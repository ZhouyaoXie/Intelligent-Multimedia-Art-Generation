import os
import psutil
import torch
import numpy as np
import socket

curr_dir = os.path.dirname(os.path.abspath(__file__))
print('current directory %s' % curr_dir)
os.chdir(curr_dir)

from base import logging

from base.global_manager import global_manager
from deepscale.pt.deepscale_model_utils import inflate_engine
from deepscale.pt.deepscale_constants import *
from ds.code.deepscale_params import DeepScaleEngineParams
from utils.general_utils import AverageMeter, AverageGroupMeter

torch.manual_seed(42)


def get_model_name(params, engine):
    model_name = params.get(MODEL_NAME, engine.__class__.__name__)
    return model_name


def setup_engine(params):
    logging.info('Engine Params = {}'.format(params.params_dict))

    # Build the model graph
    engine = inflate_engine(params)

    # Get model name
    model_name = get_model_name(params=engine.raw_params, engine=engine)
    engine.logging("Model name is %s" % model_name)

    # Load or initialize model
    engine.load()

    return engine


def reap_children():
    """Used to kill leftover children"""
    # kill all children processes
    current_process = psutil.Process()
    children = current_process.children(recursive=True)
    for child in children:
        logging.info("Try killing child process with pid %d" % child.pid)
        try:
            child.kill()
            logging.info("Successfully killed child process with pid %d" % child.pid)
        except:
            logging.warning("Failed to kill child process with pid %d" % child.pid)


def dump_lr_mom(tag, engine):
    lr = engine.get_lr()
    mom = engine.get_mom()
    engine.logging('{}: lr={}, mom={}'.format(tag, lr, mom))


def _train(engine):
    params = engine.raw_params
    logging.info(f'hostname: {socket.gethostname()}')

    # Get steps parameters
    steps_per_checkpoint = params.get(
        STEPS_PER_CHECKPOINT, STEPS_PER_CHECKPOINT_DEFAULT
    )
    steps_per_print = params.get(
        STEPS_PER_PRINT, STEPS_PER_PRINT_DEFAULT
    )
    steps_per_validation = params.get(
        STEPS_PER_VALIDATION, -1
    )
    steps_per_test = params.get(
        STEPS_PER_TEST, -1
    )
    epochs_per_test = params.get(
        EPOCHS_PER_TEST, EPOCHS_PER_TEST_DEFAULT
    )
    idle_steps_before_termination = params.get(
        IDLE_STEPS_BEFORE_TERMINATION, IDLE_STEPS_BEFORE_TERMINATION_DEFAULT
    )
    max_epoch_count = params.get(
        MAX_EPOCH_COUNT, MAX_EPOCH_COUNT_DEFAULT)

    engine.logging("%s: %s" % (STEPS_PER_CHECKPOINT, steps_per_checkpoint))
    engine.logging("%s: %s" % (STEPS_PER_PRINT, steps_per_print))
    engine.logging("%s: %s" % (STEPS_PER_VALIDATION, steps_per_validation))
    if steps_per_validation == -1:
        engine.logging("%s: disabled" % STEPS_PER_VALIDATION)
    engine.logging("%s: %s" % (STEPS_PER_TEST, steps_per_test))
    if STEPS_PER_TEST == -1:
        engine.logging("%s: disabled" % STEPS_PER_TEST)
    engine.logging("%s: %s" % (EPOCHS_PER_TEST, epochs_per_test))
    engine.logging("%s: %s" % (IDLE_STEPS_BEFORE_TERMINATION, idle_steps_before_termination))
    engine.logging("%s: %s" % (MAX_EPOCH_COUNT, max_epoch_count))

    num_worker_devices = engine.world_size

    best_test_result = None

    window_size = params['steps_per_print'] * params['gradient_accumulation_steps']

    # Initialize idle steps
    idle_steps = 0

    # Model checkpoint related
    model_name = get_model_name(params=params, engine=engine)
    engine.logging("Checkpoints saved with '{}' prefix".format(model_name))

    if params['eval_at_first']:
        best_test_result = engine.test_wrapper()
        report_eval_status(engine.params, best_test_result, best_test_result)

    while True:
        meter = {'loss': AverageMeter("loss", window_size)}
        task_meter = AverageGroupMeter([], window_size)

        engine.logging("Starting epoch %d ..." % global_manager.epoch_count)
        train_dataset = engine.deepscale_io_dataset(route=ROUTE_TRAIN)

        for batch in train_dataset:
            loss = engine.train_on_batch(batch)
            meter['loss'].update(loss['loss'])
            task_meter.update(loss)

            if global_manager.is_gradient_boundary():
                # Print train loss if steps_per_print is reached
                if global_manager.step_count % steps_per_print == 0:
                    report_status(engine.params, engine, avg_loss=meter['loss'], task_loss=task_meter)

                # Test if steps_per_test is reached
                if steps_per_test > 0 and (global_manager.step_count + 1) % (steps_per_test // num_worker_devices) == 0:
                    test_result = engine.test_wrapper()
                    report_eval_status(engine.params, test_result, best_test_result)

                    if best_test_result is None or engine.test_result_is_better(best_test_result, test_result):
                        best_test_result = test_result
                        engine.logging("Saving new best test model at step %d/%d ... under path %s" %
                                       (global_manager.epoch_count, global_manager.step_count + 1, engine.output_dir))
                        engine.save(engine.output_dir, ckpt_name=f'{model_name}.Best.pth')

                # Checkpoints if steps_per_checkpoint is reached
                if (global_manager.step_count + 1) % (steps_per_checkpoint // num_worker_devices) == 0:
                    engine.logging("Saving checkpoint model at %d/%d ..." %
                                   (global_manager.epoch_count, global_manager.step_count))
                    engine.save(engine.output_dir, ckpt_name=f'{model_name}_{global_manager.global_steps}.pth')

                if not (np.isnan(loss['loss']) or np.isinf(loss['loss'])):
                    global_manager.global_steps += 1
                    global_manager.step_count += 1
                else:
                    global_manager.skip_steps += 1
            global_manager.micro_steps += 1

        # Exits if idle_steps_before_termination is reached
        if idle_steps > idle_steps_before_termination:
            engine.logging("Idle steps limit %d reached. Exiting..." % idle_steps_before_termination)
            engine.close()
            break

        # Exits if max_epoch_count is reached
        if global_manager.epoch_count >= max_epoch_count:
            engine.logging("Reached max epoch count %d, exiting ..." % max_epoch_count)
            break

        logging.info(f'epoch {global_manager.epoch_count} is done')
        test_result = engine.test_wrapper()
        report_eval_status(engine.params, test_result, best_test_result)

        if best_test_result is None or engine.test_result_is_better(best_test_result, test_result):
            best_test_result = test_result
            engine.logging("Saving new best test model at %d/%d ..." %
                           (global_manager.epoch_count, global_manager.step_count + 1))
            engine.save(engine.output_dir, ckpt_name=f'{model_name}.Best.pth')

        global_manager.update_epoch()


def report_status(params, engine, **kwmeters):
    lr = format(engine.get_lr()[0], 'e')
    num_worker_devices = engine.world_size
    batch_size_per_device = params.batch_size
    gradient_accumulation_steps = params.gradient_accumulation_steps

    epoch_count = global_manager.epoch_count
    step_count = global_manager.step_count
    global_steps = global_manager.global_steps

    logging.info(
        '[%d]%d, lr=%s, Loss=%.5f, WindowLoss=%.5f, TaskLoss=%s, NumBatches=%s, NumSamples=%s' % (
            epoch_count + 1,
            step_count,
            lr,
            kwmeters['avg_loss'].avg,
            kwmeters['avg_loss'].window_avg,
            kwmeters['task_loss'].window_avg_str,
            step_count * num_worker_devices,
            step_count * num_worker_devices * batch_size_per_device * gradient_accumulation_steps))

    if params.tensorboard:
        logging.add_others('Others/lr', engine.get_lr()[0], global_steps)
        logging.add_losses(kwmeters['task_loss'].window_avg, global_steps)
        logging.add_losses({'train_window_loss': kwmeters['avg_loss'].window_avg,
                            'train_avg_loss': kwmeters['avg_loss'].avg},
                           global_steps, is_training=True)


def report_eval_status(params, test_result, best_test_result):
    logging.info(f'Current best result is {best_test_result}')
    logging.info(f"Test result at {global_manager.epoch_count}/{global_manager.step_count}: {test_result}")

    if params.tensorboard:
        logging.add_metrics(test_result, global_manager.global_steps)


def eval(engine):
    engine.logging("Start validating ...")
    # Note: validation_loss can be a dictionary
    test_result = engine.test_wrapper()
    engine.logging("ValidationResult=%s" % test_result)


def predict(engine):
    engine.logging("Start testing ...")
    engine.inference_wrapper()
    engine.logging("Inference done")


def encode(engine):
    engine.warning("Encode is to be implemented.")


def train():
    # Set logging level to INFO
    params_wrapper = DeepScaleEngineParams()
    params = params_wrapper.create_params_obj()
    engine = setup_engine(params)

    mode = params.mode.lower()
    if mode == ROUTE_TRAIN:
        _train(engine)
    elif mode == ROUTE_EVAL:
        eval(engine)
    elif mode == ROUTE_PREDICT:
        predict(engine)
    elif mode == ROUTE_ENCODE:
        encode(engine)
    else:
        raise ValueError("Unrecognized mode!")


if __name__ == '__main__':
    try:
        train()
    except Exception as e:
        logging.exception('error is recorded')
        raise e
