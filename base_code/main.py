import os
import logging
from parse_args import parse_arguments
from load_data import build_splits_baseline, build_splits_domain_disentangle, build_splits_clip_disentangle, build_splits_domain_generalization_clip, get_target_data
from experiments.baseline import BaselineExperiment
from experiments.domain_disentangle import DomainDisentangleExperiment
from experiments.clip_disentangle import CLIPDisentangleExperiment
from utilities import plot_losses


def setup_experiment(opt):

    if opt['experiment'] == 'baseline':
        experiment = BaselineExperiment(opt)
        train_loader, validation_loader, test_loader = build_splits_baseline(
            opt)

    elif opt['experiment'] == 'domain_disentangle':
        experiment = DomainDisentangleExperiment(opt)
        train_loader, validation_loader, test_loader = build_splits_domain_disentangle(
            opt)

    elif opt['experiment'] == 'clip_disentangle':
        experiment = CLIPDisentangleExperiment(opt)
        if not opt['domain_generalization']:
            train_loader, validation_loader, test_loader = build_splits_clip_disentangle(
                opt)
        else:
            train_loader, validation_loader, test_loader = build_splits_domain_generalization_clip(
                opt)

    else:
        raise ValueError('Experiment not yet supported.')

    print("Dataset loaded...")
    return experiment, train_loader, validation_loader, test_loader


def new_logger():
    return {
        'loss_log_train': 0,
        'loss_log_val': 0,
        'train_counter': 0,
        'val_counter': 0,
    }


def main(opt):
    experiment, train_loader, validation_loader, test_loader = setup_experiment(
        opt)

    loss_acc_logger = new_logger()
    train_losses = []
    val_losses = []
    val_accuracies = []

    if not opt['test']:  # Skip training if '--test' flag is set
        iteration = 0
        best_accuracy = 0
        total_train_loss = 0

        # Restore last checkpoint
        if os.path.exists(f'{opt["output_path"]}/last_checkpoint.pth'):
            iteration, best_accuracy, total_train_loss = experiment.load_checkpoint(
                f'{opt["output_path"]}/last_checkpoint.pth')
        else:
            logging.info(opt)

        # Train loop
        while iteration < opt['max_iterations']:
            for data in train_loader:

                # Plot tSNE features only for domain disentanglement (with or w/o Clip)
                if (not opt['experiment'] == 'baseline' and not opt['domain_generalization']) and iteration % 1000 == 0:
                    branch = -1 if iteration == 0 else 0
                    experiment.tSNE_plot(
                        train_loader, extract_features_branch=branch, iter=iteration, base_path=opt['output_path'])

                total_train_loss += experiment.train_iteration(
                    data, loss_acc_logger)

                if iteration % opt['print_every'] == 0:
                    logging.info(
                        f'[TRAIN - {iteration}] Loss: {total_train_loss / (iteration + 1)}')

                if iteration % opt['validate_every'] == 0:
                    # Run validation
                    val_accuracy, val_loss = experiment.validate(
                        validation_loader, loss_acc_logger)
                    logging.info(
                        f'[VAL - {iteration}] Loss: {val_loss} | Accuracy: {(100 * val_accuracy):.2f}')
                    if val_accuracy > best_accuracy and iteration > (opt["max_iterations"] // 2):
                        best_accuracy = val_accuracy
                        experiment.save_checkpoint(
                            f'{opt["output_path"]}/best_checkpoint.pth', iteration, best_accuracy, total_train_loss)
                    experiment.save_checkpoint(
                        f'{opt["output_path"]}/last_checkpoint.pth', iteration, best_accuracy, total_train_loss)

                    train_losses.append(
                        loss_acc_logger['loss_log_train'].item() / loss_acc_logger['train_counter'])
                    val_losses.append(
                        loss_acc_logger['loss_log_val'].item() / loss_acc_logger['val_counter'])
                    val_accuracies.append(val_accuracy)

                iteration += 1
                if iteration > opt['max_iterations']:
                    break
                loss_acc_logger = new_logger()

        print("train_losses", train_losses)
        print("val_losses", val_losses)
        print("val_accuracies", val_accuracies)
        plot_losses(train_losses, val_losses, val_accuracies, opt)

    # Test on last
    experiment.load_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth')

    # Plot tSNE features only for domain disentanglement (with or w/o Clip)
    if (not opt['experiment'] == 'baseline' and not opt['domain_generalization']):
        experiment.tSNE_plot(train_loader, extract_features_branch=0, iter='final_last',
                             base_path=opt['output_path'])

    if not opt['domain_generalization'] or opt['experiment'] == 'baseline':
        test_accuracy, _ = experiment.validate(
            test_loader, loss_acc_logger, test=True)
        logging.info(f'[TEST LAST] Accuracy: {(100 * test_accuracy):.2f}')

    # Test on Target Only if not baseline
    if not opt['experiment'] == 'baseline':
        test_tgt_only_loader = get_target_data(opt)
        test_accuracy, _ = experiment.test_on_target(test_tgt_only_loader)
        logging.info(f'[TEST TARGET LAST] Accuracy: {(100 * test_accuracy):.2f}')

    # Test on best
    experiment.load_checkpoint(f'{opt["output_path"]}/best_checkpoint.pth')

    # Plot tSNE features only for domain disentanglement (with or w/o Clip)
    if (not opt['experiment'] == 'baseline' and not opt['domain_generalization']):
        experiment.tSNE_plot(train_loader, extract_features_branch=0, iter='final_best',
                             base_path=opt['output_path'])

    if not opt['domain_generalization'] or opt['experiment'] == 'baseline':
        test_accuracy, _ = experiment.validate(
            test_loader, loss_acc_logger, test=True)
        logging.info(f'[TEST BEST] Accuracy: {(100 * test_accuracy):.2f}')

    # Test on Target Only if not baseline
    if not opt['experiment'] == 'baseline':
        test_tgt_only_loader = get_target_data(opt)
        test_accuracy, _ = experiment.test_on_target(test_tgt_only_loader)
        logging.info(f'[TEST TARGET BEST] Accuracy: {(100 * test_accuracy):.2f}')


if __name__ == '__main__':

    opt = parse_arguments()

    # Setup output directories
    os.makedirs(opt['output_path'], exist_ok=True)

    # Setup logger
    logging.basicConfig(filename=f'{opt["output_path"]}/log.txt',
                        format='%(message)s', level=logging.INFO, filemode='a')

    main(opt)
