from urllib.parse import urlparse
import torch
import pickle
from scipy.spatial import distance
import numpy as np 
import sys 
import os 
import json
import logging
import os
import shutil
import tempfile
from functools import wraps
from hashlib import sha256
import sys
from io import open

import boto3
import requests
from botocore.exceptions import ClientError
from tqdm import tqdm

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

try:
    from pathlib import Path
    PYTORCH_PRETRAINED_BERT_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',
                                                   Path.home() / '.pytorch_pretrained_bert'))
except (AttributeError, ImportError):
    PYTORCH_PRETRAINED_BERT_CACHE = os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',
                                              os.path.join(os.path.expanduser("~"), '.pytorch_pretrained_bert'))

###########################################
# little helpers
###########################################
def word2event(word_seq, idx2event):
  return [ idx2event[w] for w in word_seq ]

def get_beat_idx(event):
  return int(event.split('_')[-1])

###########################################
# sampling utilities
###########################################
def temperatured_softmax(logits, temperature):
  try:
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    assert np.count_nonzero(np.isnan(probs)) == 0
  except:
    print ('overflow detected, use 128-bit')
    logits = logits.astype(np.float128)
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    probs = probs.astype(float)
  return probs

def nucleus(probs, p):
    probs /= sum(probs)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cusum_sorted_probs > p
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][1]
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:3] # just assign a value
    candi_probs = np.array([probs[i] for i in candi_index], dtype=np.float64)
    candi_probs /= sum(candi_probs)
    word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return word

def numpy_to_tensor(arr, use_gpu=True, device='cuda:0'):
  if use_gpu:
    return torch.tensor(arr).to(device).float()
  else:
    return torch.tensor(arr).float()

def tensor_to_numpy(tensor):
  return tensor.cpu().detach().numpy()

def pickle_load(f):
  return pickle.load(open(f, 'rb'))

def pickle_dump(obj, f):
  pickle.dump(obj, open(f, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


def cached_path(url_or_filename, cache_dir=None):
  """
  Given something that might be a URL (or might be a local path),
  determine which. If it's a URL, download the file and cache it, and
  return the path to the cached file. If it's already a local path,
  make sure the file exists and then return the path.
  """
  if cache_dir is None:
      cache_dir = PYTORCH_PRETRAINED_BERT_CACHE
  if sys.version_info[0] == 3 and isinstance(url_or_filename, Path):
      url_or_filename = str(url_or_filename)
  if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
      cache_dir = str(cache_dir)

  parsed = urlparse(url_or_filename)

  if parsed.scheme in ('http', 'https', 's3'):
      # URL, so get it from the cache (downloading if necessary)
      return get_from_cache(url_or_filename, cache_dir)
  elif os.path.exists(url_or_filename):
      # File, and it exists.
      return url_or_filename
  elif parsed.scheme == '':
      # File, but it doesn't exist.
      raise EnvironmentError("file {} not found".format(url_or_filename))
  else:
      # Something unknown
      raise ValueError("unable to parse {} as a URL or as a local path".format(url_or_filename))


def get_from_cache(url, cache_dir=None):
  """
  Given a URL, look for the corresponding dataset in the local cache.
  If it's not there, download it. Then return the path to the cached file.
  """
  if cache_dir is None:
      cache_dir = PYTORCH_PRETRAINED_BERT_CACHE
  if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
      cache_dir = str(cache_dir)

  if not os.path.exists(cache_dir):
      os.makedirs(cache_dir)

  # Get eTag to add to filename, if it exists.
  if url.startswith("s3://"):
      etag = s3_etag(url)
  else:
      response = requests.head(url, allow_redirects=True)
      if response.status_code != 200:
          raise IOError("HEAD request failed for url {} with status code {}"
                        .format(url, response.status_code))
      etag = response.headers.get("ETag")

  filename = url_to_filename(url, etag)

  # get cache path to put the file
  cache_path = os.path.join(cache_dir, filename)

  if not os.path.exists(cache_path):
      # Download to temporary file, then copy to cache dir once finished.
      # Otherwise you get corrupt cache entries if the download gets interrupted.
      with tempfile.NamedTemporaryFile() as temp_file:
          logger.info("%s not found in cache, downloading to %s", url, temp_file.name)

          # GET file object
          if url.startswith("s3://"):
              s3_get(url, temp_file)
          else:
              http_get(url, temp_file)

          # we are copying the file before closing it, so flush to avoid truncation
          temp_file.flush()
          # shutil.copyfileobj() starts at the current position, so go to the start
          temp_file.seek(0)

          logger.info("copying %s to cache at %s", temp_file.name, cache_path)
          with open(cache_path, 'wb') as cache_file:
              shutil.copyfileobj(temp_file, cache_file)

          logger.info("creating metadata file for %s", cache_path)
          meta = {'url': url, 'etag': etag}
          meta_path = cache_path + '.json'
          with open(meta_path, 'w', encoding="utf-8") as meta_file:
              json.dump(meta, meta_file)

          logger.info("removing temp file %s", temp_file.name)

  return cache_path

def url_to_filename(url, etag=None):
    """
    Convert `url` into a hashed filename in a repeatable way.
    If `etag` is specified, append its hash to the url's, delimited
    by a period.
    """
    url_bytes = url.encode('utf-8')
    url_hash = sha256(url_bytes)
    filename = url_hash.hexdigest()

    if etag:
        etag_bytes = etag.encode('utf-8')
        etag_hash = sha256(etag_bytes)
        filename += '.' + etag_hash.hexdigest()

    return filename


def split_s3_path(url):
    """Split a full s3 path into the bucket name and path."""
    parsed = urlparse(url)
    if not parsed.netloc or not parsed.path:
        raise ValueError("bad s3 path {}".format(url))
    bucket_name = parsed.netloc
    s3_path = parsed.path
    # Remove '/' at beginning of path.
    if s3_path.startswith("/"):
        s3_path = s3_path[1:]
    return bucket_name, s3_path


def s3_request(func):
    """
    Wrapper function for s3 requests in order to create more helpful error
    messages.
    """

    @wraps(func)
    def wrapper(url, *args, **kwargs):
        try:
            return func(url, *args, **kwargs)
        except ClientError as exc:
            if int(exc.response["Error"]["Code"]) == 404:
                raise EnvironmentError("file {} not found".format(url))
            else:
                raise

    return wrapper


@s3_request
def s3_etag(url):
    """Check ETag on S3 object."""
    s3_resource = boto3.resource("s3")
    bucket_name, s3_path = split_s3_path(url)
    s3_object = s3_resource.Object(bucket_name, s3_path)
    return s3_object.e_tag


@s3_request
def s3_get(url, temp_file):
    """Pull a file directly from S3."""
    s3_resource = boto3.resource("s3")
    bucket_name, s3_path = split_s3_path(url)
    s3_resource.Bucket(bucket_name).download_fileobj(s3_path, temp_file)


def http_get(url, temp_file):
    req = requests.get(url, stream=True)
    content_length = req.headers.get('Content-Length')
    total = int(content_length) if content_length is not None else None
    progress = tqdm(unit="B", total=total)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk: # filter out keep-alive new chunks
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()


def get_from_cache(url, cache_dir=None):
    """
    Given a URL, look for the corresponding dataset in the local cache.
    If it's not there, download it. Then return the path to the cached file.
    """
    if cache_dir is None:
        cache_dir = PYTORCH_PRETRAINED_BERT_CACHE
    if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Get eTag to add to filename, if it exists.
    if url.startswith("s3://"):
        etag = s3_etag(url)
    else:
        response = requests.head(url, allow_redirects=True)
        if response.status_code != 200:
            raise IOError("HEAD request failed for url {} with status code {}"
                          .format(url, response.status_code))
        etag = response.headers.get("ETag")

    filename = url_to_filename(url, etag)

    # get cache path to put the file
    cache_path = os.path.join(cache_dir, filename)

    if not os.path.exists(cache_path):
        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        with tempfile.NamedTemporaryFile() as temp_file:
            logger.info("%s not found in cache, downloading to %s", url, temp_file.name)

            # GET file object
            if url.startswith("s3://"):
                s3_get(url, temp_file)
            else:
                http_get(url, temp_file)

            # we are copying the file before closing it, so flush to avoid truncation
            temp_file.flush()
            # shutil.copyfileobj() starts at the current position, so go to the start
            temp_file.seek(0)

            logger.info("copying %s to cache at %s", temp_file.name, cache_path)
            with open(cache_path, 'wb') as cache_file:
                shutil.copyfileobj(temp_file, cache_file)

            logger.info("creating metadata file for %s", cache_path)
            meta = {'url': url, 'etag': etag}
            meta_path = cache_path + '.json'
            with open(meta_path, 'w', encoding="utf-8") as meta_file:
                json.dump(meta, meta_file)

            logger.info("removing temp file %s", temp_file.name)

    return cache_path


def read_set_from_file(filename):
    '''
    Extract a de-duped collection (set) of text from a file.
    Expected file format is one item per line.
    '''
    collection = set()
    with open(filename, 'r', encoding='utf-8') as file_:
        for line in file_:
            collection.add(line.rstrip())
    return collection


def get_file_extension(path, dot=True, lower=True):
    ext = os.path.splitext(path)[1]
    ext = ext if dot else ext[1:]
    return ext.lower() if lower else ext