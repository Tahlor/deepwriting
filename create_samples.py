import tensorflow as tf
import numpy as np

import sys
sys.path.append(r"./source")

import os
import argparse
import json
from scipy.misc import imsave

from tf_dataset_hw import *
from tf_models import VRNNGMM
from tf_models_hw import HandwritingVRNNGmmModel, HandwritingVRNNModel
from utils_visualization import plot_latent_variables, plot_latent_categorical_variables, plot_matrix_and_get_image, plot_and_get_image
import visualize_hw as visualize

# Sampling options
run_gmm_eval = False  # Visualize GMM latent space by using random samples and T-SNE.
run_original_sample = True  # Save an image of reference samples (see reference_sample_ids).
run_reconstruction = False  # Reconstruct reference samples and save reconstruction results.
run_biased_sampling = False  # Use a real reference sample to infer style (see reference_sample_ids) and synthesize the given text (see conditional_texts).
run_unbiased_sampling = True  # Use a random style to synthesize the given text (see conditional_texts).
run_colored_png_output = False  # Save colored images (see line 47). For now we use end-of-character probabilities to assign new colors.

# Sampling hyper-parameters
eoc_threshold = 0.05
cursive_threshold = 0.005
ref_len = None  # Use the whole sequence.
seq_len = 800  # Maximum number of steps.
gmm_num_samples = 500  # For run_gmm_eval only.

# Text to be written by the model.
conditional_texts = ["I am a synthetic sample", "I can write this line in any style."]*10 # doesn't work with double spaces!!!
# Indices of reference style samples from validation split.
reference_sample_ids = [107, 226, 696]
# Concatenate reference sample with synthetic sample to make a direct comparison.
concat_ref_and_synthetic_samples = False

# Sampling output options
plot_eoc = False  # Plot end-of-character probabilities.
plot_latent_vars = False  # Plot a matrix of approximate posterior and prior mu values.
save_plots = True  # Save plots as image.
show_plots = False  # Show plots in a window.

class ImageGenerator:

    def __init__(self, config, verbose=0):
        self.sess = None
        self.model = None
        self.validation = None
        self.config = config
        self.keyword_args = dict()
        #self.keyword_args['conditional_inputs'] = None
        self.keyword_args['eoc_threshold'] = eoc_threshold
        self.keyword_args['cursive_threshold'] = cursive_threshold
        self.keyword_args['use_sample_mean'] = True
        self.sess, self.model, self.validation_dataset = self.load_model(self.config)

    @staticmethod
    def plot_eval_details(data_dict, sample, save_dir, save_name):
        visualize.draw_stroke_svg(sample, factor=0.001, svg_filename=os.path.join(save_dir, save_name + '.svg'))

        plot_data = {}
        if run_colored_png_output:
            synthetic_eoc = np.squeeze(data_dict['out_eoc'])
            visualize.draw_stroke_svg(sample, factor=0.001, color_labels=synthetic_eoc > eoc_threshold,
                                      svg_filename=os.path.join(save_dir, save_name + '_colored.svg'))

        if plot_latent_vars and 'p_mu' in data_dict:
            plot_data['p_mu'] = np.transpose(data_dict['p_mu'][0], [1, 0])
            plot_data['q_mu'] = np.transpose(data_dict['q_mu'][0], [1, 0])
            plot_data['q_sigma'] = np.transpose(data_dict['q_sigma'][0], [1, 0])
            plot_data['p_sigma'] = np.transpose(data_dict['p_sigma'][0], [1, 0])

            plot_img = plot_latent_variables(plot_data, show_plot=show_plots)
            if save_plots:
                imsave(os.path.join(save_dir, save_name + '_normal.png'), plot_img)

        if plot_latent_vars and 'p_pi' in data_dict:
            plot_data['p_pi'] = np.transpose(data_dict['p_pi'][0], [1, 0])
            plot_data['q_pi'] = np.transpose(data_dict['q_pi'][0], [1, 0])
            plot_img = plot_latent_categorical_variables(plot_data, show_plot=show_plots)
            if save_plots:
                imsave(os.path.join(save_dir, save_name + '_pi.png'), plot_img)

        if plot_eoc and 'out_eoc' in data_dict:
            plot_img = plot_and_get_image(np.squeeze(data_dict['out_eoc']))
            imsave(os.path.join(save_dir, save_name + '_eoc.png'), plot_img)

        # Same for every sample.
        if 'gmm_mu' in data_dict:
            gmm_mu_img = plot_matrix_and_get_image(data_dict['gmm_mu'])
            gmm_sigma_img = plot_matrix_and_get_image(data_dict['gmm_sigma'])
            if save_plots:
                imsave(os.path.join(save_dir, 'gmm_mu.png'), gmm_mu_img)
                imsave(os.path.join(save_dir, 'gmm_sigma.png'), gmm_sigma_img)

        return plot_data

    @staticmethod
    def load_model(config):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        Model_cls = getattr(sys.modules[__name__], config['model_cls'])
        Dataset_cls = getattr(sys.modules[__name__], config['dataset_cls'])

        batch_size = 1
        data_sequence_length = None
        # Load validation dataset to fetch statistics.
        if issubclass(Dataset_cls, HandWritingDatasetConditional):
            validation_dataset = Dataset_cls(config['validation_data'], var_len_seq=True, use_bow_labels=config['use_bow_labels'])
        elif issubclass(Dataset_cls, HandWritingDataset):
            validation_dataset = Dataset_cls(config['validation_data'], var_len_seq=True)
        else:
            raise Exception("Unknown dataset class.")

        strokes = tf.placeholder(tf.float32, shape=[batch_size, data_sequence_length, sum(validation_dataset.input_dims)])
        targets = tf.placeholder(tf.float32, shape=[batch_size, data_sequence_length, sum(validation_dataset.target_dims)])
        sequence_length = tf.placeholder(tf.int32, shape=[batch_size])

        # Create inference graph.
        with tf.name_scope("validation"):
            inference_model = Model_cls(config,
                                        reuse=False,
                                        input_op=strokes,
                                        target_op=targets,
                                        input_seq_length_op=sequence_length,
                                        input_dims=validation_dataset.input_dims,
                                        target_dims=validation_dataset.target_dims,
                                        batch_size=batch_size,
                                        mode="validation",
                                        data_processor=validation_dataset)
            inference_model.build_graph()
            inference_model.create_image_summary(validation_dataset.prepare_for_visualization)

        # Create sampling graph.
        with tf.name_scope("sampling"):
            model = Model_cls(config,
                              reuse=True,
                              input_op=strokes,
                              target_op=None,
                              input_seq_length_op=sequence_length,
                              input_dims=validation_dataset.input_dims,
                              target_dims=validation_dataset.target_dims,
                              batch_size=batch_size,
                              mode="sampling",
                              data_processor=validation_dataset)
            model.build_graph()

        # Create a session object and initialize parameters.
        #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        sess = tf.Session()

        # Restore computation graph.
        try:
            saver = tf.train.Saver()
            # Restore variables.
            if config['checkpoint_id'] is None:
                checkpoint_path = tf.train.latest_checkpoint(config['model_dir'])
            else:
                checkpoint_path = os.path.join(config['model_dir'], config['checkpoint_id'])

            print("Loading model " + checkpoint_path)
            saver.restore(sess, checkpoint_path)
        except:
            raise Exception("Model is not found.")

        return sess, model, validation_dataset


    def close(self):
        self.sess.close()

    def random_sample(self):
        # Conditional handwriting synthesis.
        for text_id, text in enumerate(conditional_texts):
            self.keyword_args['use_sample_mean'] = True # disable beautification

            print(self.keyword_args, seq_len, text)
            unbiased_sampling_results = self.model.sample_unbiased(session=self.sess, seq_len=seq_len, conditional_inputs=text, **self.keyword_args)

            save_name = 'synthetic_unbiased_(' + str(text_id) + ')'
            synthetic_sample = self.validation_dataset.undo_normalization(unbiased_sampling_results[0]['output_sample'][0],
                                                                     detrend_sample=False)

            self.plot_eval_details(unbiased_sampling_results[0], synthetic_sample, self.config['eval_dir'], save_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-S', '--model_save_dir', dest='model_save_dir', type=str, default='./runs/', help='path to main model save directory')
    parser.add_argument('-E', '--eval_dir', type=str, default='./runs_evaluation/', help='path to main log/output directory')
    parser.add_argument('-M', '--model_id', dest='model_id', type=str, help='model folder')
    parser.add_argument('-C', '--checkpoint_id', type=str, default=None, help='log and output directory')
    parser.add_argument('-QN', '--quantitative', dest='quantitative', action="store_true", help='Run quantitative analysis')
    parser.add_argument('-QL', '--qualitative', dest='qualitative', action="store_true", help='Run qualitative analysis')
    parser.add_argument('-V', '--verbose', dest='verbose', type=int, default=1, help='Verbosity')
    args=parser.parse_args()
    config_dict = json.load(open(os.path.abspath(os.path.join(args.model_save_dir, args.model_id, 'config.json')), 'r'))
    config_dict['model_dir'] = os.path.join(args.model_save_dir, args.model_id)  # in case the folder is renamed.
    config_dict['checkpoint_id'] = args.checkpoint_id
    config_dict['batch_size'] = 16
    if args.eval_dir is None:
        config_dict['eval_dir'] = config_dict['model_dir']
    else:
        config_dict['eval_dir'] = os.path.join(args.eval_dir, args.model_id)

    if not os.path.exists(config_dict['eval_dir']):
        os.makedirs(config_dict['eval_dir'])

    gen = ImageGenerator(config=config_dict, verbose=args.verbose)
    gen.random_sample()
