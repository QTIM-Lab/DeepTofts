from qtim_tools.qtim_dce.dce_util import generate_AIF, parker_model_AIF, convert_intensity_to_concentration, revert_concentration_to_intensity, estimate_concentration, estimate_concentration_general

from qtim_tools.qtim_utilities.format_util import convert_input_2_numpy
from qtim_tools.qtim_utilities.nifti_util import save_numpy_2_nifti

import generate_data
import random
import numpy as np
import math
import glob
import dicom
import os
import scipy

class SequenceData(object):

    def __init__(self):
        pass

    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """

        if self.batch_id == len(self.data):
            self.batch_id = 0

        batch_data = (self.data[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id + batch_size, len(self.data))])

        self.batch_id = min(self.batch_id + batch_size, len(self.data))

        return batch_data, batch_labels, batch_seqlen

    def save(self, output_filepaths):

        for data, idx in enumerate([self.data, self.labels, self.seqlen]):
            np.save(output_filepaths[idx], data)

    def load(self, input_filepaths):

        self.data = np.load(input_filepaths[0])
        self.labels = np.load(input_filepaths[1])
        self.seqlen = np.load(input_filepaths[2])
        self.batch_id = 0

class DCESequenceData(SequenceData):

    def __init__(self, dce_data, ktrans_data, ve_data, n_samples=1000, gaussian_noise=[0,0], reconstruct=False, overwrite=False, masked=True):

        self.data = []
        self.labels = []
        self.seqlen = []

        self.completed = False

        dce_raw, ve_raw, ktrans_raw = convert_input_2_numpy(dce_data), convert_input_2_numpy(ve_data), convert_input_2_numpy(ktrans_data)

        # Minit Test
        # dce_raw, ve_raw, ktrans_raw = dce_raw[0:30,0:30,0:30,:], ve_raw[0:30,0:30,0:30], ktrans_raw[0:30,0:30,0:30]

        dce_numpy = dce_raw.reshape(-1, dce_raw.shape[-1])
        ve_numpy, ktrans_numpy = ve_raw.reshape(ve_raw.shape[0]*ve_raw.shape[1]*ve_raw.shape[2]), ktrans_raw.reshape(ktrans_raw.shape[0]*ktrans_raw.shape[1]*ktrans_raw.shape[2])

        self.dce_data = [dce_numpy, ktrans_numpy, ve_numpy]
        self.data_shape = ktrans_raw.shape
        self.voxel_count = ktrans_numpy.size

        dce_seq_len = dce_numpy.shape[-1]
        dce_idx = np.arange(self.voxel_count)

        if masked:
            ktrans_high_mask = ktrans_numpy > .06
            ktrans_low_mask = ktrans_numpy <=.06
            dce_idx = dce_idx[ktrans_low_mask][0:int(np.ceil(2*n_samples/9))].tolist() + dce_idx[ktrans_high_mask][0:int(np.ceil(7*n_samples/9))+1].tolist()

        # print np.min(ktrans_numpy), np.max(ktrans_numpy)

        for i in range(n_samples):

            # Random sequence length
            seq_len = dce_seq_len
            
            # Monitor sequence length for TensorFlow dynamic calculation
            self.seqlen.append(seq_len)
                
            Intensity = dce_numpy[dce_idx[i], :]
            ktrans = ktrans_numpy[dce_idx[i]]
            ve = ve_numpy[dce_idx[i]]

            s = Intensity
            s = [[i] for i in s]

            self.data.append(s)
            self.labels.append([ktrans, ve])


        self.batch_id = 0    

class ToftsPhantomReconstructionData(SequenceData):

    def __init__(self, phantom_data, gaussian_noise=[0,0]):

        self.completed = False

        dce_raw = convert_input_2_numpy(phantom_data)

        self.aif = np.mean(dce_raw[:, 70:, :], axis=(0,1))
        self.data_shape = dce_raw[...,0].shape

        dce_numpy = dce_raw.reshape(-1, dce_raw.shape[-1])

        self.dce_data = dce_numpy
        self.voxel_count = dce_numpy[...,0].size

        self.batch_id = 0

    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """

        if self.batch_id == self.voxel_count:
            self.batch_id = 0
            self.completed = True

        batch_data = []
        for i in range(self.batch_id, min(self.batch_id + batch_size, self.voxel_count)):
            temp_data = self.dce_data[i,:]
            batch_data.append([[temp_data[i], self.aif[i]] for i in xrange(temp_data.size)] + [[0.,0] for i in range(65 - temp_data.size)])

        batch_seqlen = [self.dce_data.shape[-1]] *len(batch_data)

        self.batch_id = min(self.batch_id + batch_size, self.voxel_count)

        return batch_data, batch_seqlen

    def save(self, output_filepaths):

        for data, idx in enumerate([self.dce_numpy, self.voxel_count, self.data_shape]):
            np.save(output_filepaths[idx], data)

    def load(self, input_filepaths):

        self.dce_numpy = np.load(input_filepaths[0])
        self.voxel_count = np.load(input_filepaths[1])
        self.data_shape = np.load(input_filepaths[2])

class DCEReconstructionData(SequenceData):

    def __init__(self, dce_data, ktrans_data, ve_data, n_samples=1000, gaussian_noise=[0,0]):

        self.completed = False

        dce_raw, ve_raw, ktrans_raw = convert_input_2_numpy(dce_data), convert_input_2_numpy(ve_data), convert_input_2_numpy(ktrans_data)

        # Minit Test
        # dce_raw, ve_raw, ktrans_raw = dce_raw[0:30,0:30,0:30,:], ve_raw[0:30,0:30,0:30], ktrans_raw[0:30,0:30,0:30]

        dce_numpy = dce_raw.reshape(-1, dce_raw.shape[-1])
        ve_numpy, ktrans_numpy = ve_raw.reshape(ve_raw.shape[0]*ve_raw.shape[1]*ve_raw.shape[2]), ktrans_raw.reshape(ktrans_raw.shape[0]*ktrans_raw.shape[1]*ktrans_raw.shape[2])

        self.dce_data = [dce_numpy, ktrans_numpy, ve_numpy]
        self.data_shape = ktrans_raw.shape
        self.voxel_count = ktrans_numpy.size

        self.batch_id = 0

    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """

        if self.batch_id == self.voxel_count:
            self.batch_id = 0
            self.completed = True

        batch_data = []
        for i in range(self.batch_id, min(self.batch_id + batch_size, self.voxel_count)):
            temp_data = self.dce_data[0][i,:]
            batch_data.append([[i] for i in temp_data])

        batch_seqlen = [self.dce_data[0].shape[-1]] *len(batch_data)

        self.batch_id = min(self.batch_id + batch_size, self.voxel_count)

        return batch_data, batch_seqlen

    def save(self, output_filepaths):

        for data, idx in enumerate([self.dce_numpy, self.voxel_count, self.data_shape]):
            np.save(output_filepaths[idx], data)

    def load(self, input_filepaths):

        self.dce_numpy = np.load(input_filepaths[0])
        self.voxel_count = np.load(input_filepaths[1])
        self.data_shape = np.load(input_filepaths[2])
        self.batch_id = 0

class ToftsPhantomData(SequenceData):

    def __init__(self, phantom_data_files, n_samples=1000, max_seq_len=65):

        self.data = []
        self.labels = []
        self.seqlen = []

        self.completed = False

        dce_phantoms = [convert_input_2_numpy(data) for data in phantom_data_files]

        sample_per_phantom = n_samples / len(dce_phantoms)

        ktrans_values = np.array([.01, .02, .05, .1, .2, .5])
        ktrans_values = np.repeat(np.repeat(ktrans_values, 10)[:,np.newaxis], 50, axis=1).T

        ve_values = np.array([.01, .05, .1, .2, .5])
        ve_values = np.repeat(np.repeat(ve_values, 10)[:,np.newaxis], 60, axis=1)

        concentration_sample = list(np.ndindex(50, 60))
        aif_sample = list(np.ndindex(50, 10))

        indices = [(x,y) for x in concentration_sample for y in aif_sample]
        print ktrans_values.shape

        for phantom in dce_phantoms:

            seq_len = phantom.shape[-1]

            for i in range(sample_per_phantom):

                self.seqlen.append(seq_len)
                
                x, y = random.choice(indices)

                Intensity = phantom[x[0], x[1] + 10, :]
                AIF = phantom[y[0], y[1] + 70, :]

                ktrans = ktrans_values[x]
                ve = ve_values[x]

                s = []
                for idx in xrange(len(Intensity)):
                    s += [[Intensity[idx], AIF[idx]]]
                s += [[0.,0] for i in range(max_seq_len - seq_len)]

                self.data.append(s)
                self.labels.append([ktrans, ve])

        self.batch_id = 0    


class ToftsSequenceData(SequenceData):
    """ Generate sequence of data # with dynamic length.
    This class generate samples for training:
    - Class 0: Tofts sequences
    - Class 1: random sequences (i.e. [1, 3, 10, 7,...])
    NOTICE:
    We have to pad each sequence to reach 'max_seq_len' for TensorFlow
    consistency (we cannot feed a numpy array # with inconsistent
    dimensions). The dynamic calculation will then be perform thanks to
    'seqlen' attribute that records every actual sequence length.
    """
    def __init__(self, n_samples=1000, max_seq_len=60, min_seq_len=25, ktrans_range=[.001,2], ve_range=[0.01,.99], gaussian_noise=[0,0], T1_range=[1000,1000], TR_range=[5, 5], flip_angle_degrees_range=[30,30], relaxivity_range=[.0045, .0045], hematocrit_range=[.45,.45], sequence_length_range=[50,50], time_interval_seconds_range=[2,2], injection_start_time_seconds_range=[10,10], T1_blood_range=[1440,1440], baseline_intensity=[100,100], with_AIF=False):
        
        ktrans_low_range = [.001, .3]

        self.data = []
        self.labels = []
        self.seqlen = []

        for i in range(n_samples):

            # Random sequence length
            seq_len = np.random.random_integers(*sequence_length_range)
            
            # Monitor sequence length for TensorFlow dynamic calculation
            self.seqlen.append(seq_len)
            
            # Add a random or linear int sequence (50% prob)
            if random.random() < .5 or True:    
                
                injection_start_time_seconds = np.random.uniform(*injection_start_time_seconds_range)
                time_interval_seconds = np.random.uniform(*time_interval_seconds_range)
                time_interval_minutes = time_interval_seconds/60
                scan_time_seconds = seq_len * time_interval_seconds

                # Adjust for Unrealistically late injection time. Do this in a one liner later.
                while injection_start_time_seconds > .8*scan_time_seconds:
                    injection_start_time_seconds = np.random.uniform(*injection_start_time_seconds_range)

                AIF = parker_model_AIF(scan_time_seconds, injection_start_time_seconds, time_interval_seconds, timepoints=seq_len)

                ktrans = np.random.uniform(*ktrans_range)
                ve = np.random.uniform(*ve_range)

                Concentration = np.array(estimate_concentration([ktrans, ve], AIF, time_interval_minutes))

                Intensity = revert_concentration_to_intensity(data_numpy=Concentration, reference_data_numpy=[], T1_tissue=np.random.uniform(*T1_range), TR=np.random.uniform(*TR_range), flip_angle_degrees=np.random.uniform(*flip_angle_degrees_range), injection_start_time_seconds=injection_start_time_seconds, relaxivity=np.random.uniform(*relaxivity_range), time_interval_seconds=time_interval_seconds, hematocrit=np.random.uniform(*hematocrit_range), T1_blood=0, T1_map=None, static_baseline=np.random.uniform(*baseline_intensity)).tolist()

                s = Intensity

                # Normalize
                # Intensity = (Intensity - np.mean(Intensity)) / np.std(Intensity)

                s += [0. for i in range(max_seq_len - seq_len)]

                s = [[i] for i in s]

                self.data.append(s)
                self.labels.append([ktrans, ve])


        self.batch_id = 0

class ToftsPatchData(SequenceData):
    """ Generate sequence of data # with dynamic length.
    This class generate samples for training:
    - Class 0: Tofts sequences
    - Class 1: random sequences (i.e. [1, 3, 10, 7,...])
    NOTICE:
    We have to pad each sequence to reach 'max_seq_len' for TensorFlow
    consistency (we cannot feed a numpy array # with inconsistent
    dimensions). The dynamic calculation will then be perform thanks to
    'seqlen' attribute that records every actual sequence length.
    """
    def __init__(self, n_samples=1000, max_seq_len=60, min_seq_len=25, ktrans_range=[.001,2], ve_range=[0.01,.99], gaussian_noise=[0,0], T1_range=[1000,1000], TR_range=[5, 5], flip_angle_degrees_range=[30,30], relaxivity_range=[.0045, .0045], hematocrit_range=[.45,.45], sequence_length_range=[50,50], time_interval_seconds_range=[2,2], injection_start_time_seconds_range=[10,10], T1_blood_range=[1440,1440], baseline_intensity=[100,100]):
        
        ktrans_low_range = [.001, .3]

        self.data = []
        self.labels = []
        self.seqlen = []

        for i in range(n_samples):

            # Random sequence length
            seq_len = np.random.random_integers(*sequence_length_range)
            
            # Monitor sequence length for TensorFlow dynamic calculation
            self.seqlen.append(seq_len)
            
            # Add a random or linear int sequence (50% prob)
            if random.random() < .5 or True:    
                
                injection_start_time_seconds = np.random.uniform(*injection_start_time_seconds_range)
                time_interval_seconds = np.random.uniform(*time_interval_seconds_range)
                time_interval_minutes = time_interval_seconds/60
                scan_time_seconds = seq_len * time_interval_seconds

                # Adjust for Unrealistically late injection time. Do this in a one liner later.
                while injection_start_time_seconds > .8*scan_time_seconds:
                    injection_start_time_seconds = np.random.uniform(*injection_start_time_seconds_range)

                AIF = parker_model_AIF(scan_time_seconds, injection_start_time_seconds, time_interval_seconds, timepoints=seq_len)

                ktrans = np.random.uniform(*ktrans_range)
                ve = np.random.uniform(*ve_range)

                Concentration = np.array(estimate_concentration([ktrans, ve], AIF, time_interval_minutes))

                Intensity = revert_concentration_to_intensity(data_numpy=Concentration, reference_data_numpy=[], T1_tissue=np.random.uniform(*T1_range), TR=np.random.uniform(*TR_range), flip_angle_degrees=np.random.uniform(*flip_angle_degrees_range), injection_start_time_seconds=injection_start_time_seconds, relaxivity=np.random.uniform(*relaxivity_range), time_interval_seconds=time_interval_seconds, hematocrit=np.random.uniform(*hematocrit_range), T1_blood=0, T1_map=None, static_baseline=np.random.uniform(*baseline_intensity)).tolist()

                s = Intensity

                # Normalize
                # Intensity = (Intensity - np.mean(Intensity)) / np.std(Intensity)

                s += [0. for i in range(max_seq_len - seq_len)]

                s = [[i] for i in s]

                self.data.append(s)
                self.labels.append([ktrans, ve])

        self.batch_id = 0

class SineSequenceData(SequenceData):

    def __init__(self, n_samples=1000, max_seq_len=100, min_seq_len=50, amplitude_range=[1,1], period_range=[.1,1], x_shift_range=[1,1], y_shift_range=[1,1]):

        self.data = []
        self.labels = []
        self.seqlen = []

        for i in range(n_samples):

            seq_len = random.randint(min_seq_len, max_seq_len)

            # Monitor sequence length for TensorFlow dynamic calculation
            self.seqlen.append(seq_len)
            
            # Add a random or linear int sequence (50% prob)
            if random.random() < .5 or True:    
                
                amplitude = np.random.uniform(*amplitude_range)
                period = np.random.uniform(*period_range)
                x_shift = np.random.uniform(*x_shift_range)
                y_shift = np.random.uniform(*y_shift_range)

                s = [[amplitude * math.sin(period*t + x_shift) + y_shift] for t in np.arange(0, seq_len)]

                s += [[0.] for i in range(max_seq_len - seq_len)]

                self.data.append(s)
                self.labels.append([period])

                # Classification Task
                # if amplitude > 2:
                    # self.labels.append(1)
                # else:
                    # self.labels.append(0)

            else:

                pass

        # print self.labels
        # for d in self.data:
            # print d

        self.batch_id = 0

class ToySequenceData(SequenceData):
    """ Generate sequence of data with dynamic length.
    This class generate samples for training:
    - Class 0: linear sequences (i.e. [0, 1, 2, 3,...])
    - Class 1: random sequences (i.e. [1, 3, 10, 7,...])
    NOTICE:
    We have to pad each sequence to reach 'max_seq_len' for TensorFlow
    consistency (we cannot feed a numpy array with inconsistent
    dimensions). The dynamic calculation will then be perform thanks to
    'seqlen' attribute that records every actual sequence length.
    """
    
    def __init__(self, n_samples=1000, max_seq_len=20, min_seq_len=3,
                 max_value=1000):

        self.data = []
        self.labels = []
        self.seqlen = []

        for i in range(n_samples):

            # Random sequence length
            seq_len = random.randint(min_seq_len, max_seq_len)

            # Monitor sequence length for TensorFlow dynamic calculation
            self.seqlen.append(seq_len)

            # Add a random or linear int sequence (50% prob)
            if random.random() < .5:

                # Generate a linear sequence
                rand_start = random.randint(0, max_value - seq_len)
                s = [[float(i)/max_value] for i in range(rand_start, rand_start + seq_len)]

                # Pad sequence for dimension consistency
                s += [[0.] for i in range(max_seq_len - seq_len)]

                self.data.append(s)
                self.labels.append([1., 0.])

            else:

                # Generate a random sequence
                s = [[float(random.randint(0, max_value))/max_value] for i in range(seq_len)]

                # Pad sequence for dimension consistency
                s += [[0.] for i in range(max_seq_len - seq_len)]
                self.data.append(s)
                self.labels.append([0., 1.])

        # for d in self.data:
            # print d
        self.batch_id = 0

class ToyPatchData(SequenceData):
    """ Generate sequence of data with dynamic length.
    This class generate samples for training:
    - Class 0: linear sequences (i.e. [0, 1, 2, 3,...])
    - Class 1: random sequences (i.e. [1, 3, 10, 7,...])
    NOTICE:
    We have to pad each sequence to reach 'max_seq_len' for TensorFlow
    consistency (we cannot feed a numpy array with inconsistent
    dimensions). The dynamic calculation will then be perform thanks to
    'seqlen' attribute that records every actual sequence length.
    """
    
    def __init__(self, n_samples=1000, max_seq_len=10, min_seq_len=10, patch_x=5, patch_y=5, increment_range=[.02, .06]):

        self.data = []
        self.labels = []
        self.seqlen = []

        for i in range(n_samples):

            # Random sequence length
            seq_len = random.randint(min_seq_len, max_seq_len)
            increment = np.random.uniform(*increment_range)
            starting = np.random.rand(patch_x, patch_y, 1)

            s = np.zeros((patch_x, patch_y, seq_len, 1))

            # Monitor sequence length for TensorFlow dynamic calculation
            self.seqlen.append(seq_len)

            # Add a random or linear int sequence (50% prob)
            if random.random() < .5:

                # Generate a linear sequence
                s[:,:,0,:] = starting
                for t in xrange(1, seq_len):
                    s[:, :, t, :] = s[:,:,t-1,:] + increment

                self.data.append(s)
                self.labels.append([1., 0.])

            else:

                # Generate a linear sequence
                for t in xrange(seq_len):
                    s[:, :, t, :] = np.random.rand(patch_x, patch_y, 1)

                self.data.append(s)
                self.labels.append([0., 1.])

        # for d in self.data:
        #     print d
        self.batch_id = 0

class ToftsPatchData(SequenceData):
    """ Generate sequence of data # with dynamic length.
    This class generate samples for training:
    - Class 0: Tofts sequences
    - Class 1: random sequences (i.e. [1, 3, 10, 7,...])
    NOTICE:
    We have to pad each sequence to reach 'max_seq_len' for TensorFlow
    consistency (we cannot feed a numpy array # with inconsistent
    dimensions). The dynamic calculation will then be perform thanks to
    'seqlen' attribute that records every actual sequence length.
    """
    def __init__(self, n_samples=1000, max_seq_len=65, min_seq_len=56, patch_x=5, patch_y=5, ktrans_range=[.001,2], ve_range=[0.01,.99], gaussian_noise=[0,0], T1_range=[900,1500], TR_range=[3, 6], flip_angle_degrees_range=[15, 35], relaxivity_range=[.0045, .0045], hematocrit_range=[.45,.45], sequence_length_range=[65,65], time_interval_seconds_range=[1.5,6], injection_start_time_seconds_range=[8,20], T1_blood_range=[1440,1440], baseline_intensity=[20,300]):

        self.data = []
        self.labels = []
        self.seqlen = []

        for i in range(n_samples):

            # Random sequence length
            seq_len = np.random.random_integers(*sequence_length_range)
            
            # Monitor sequence length for TensorFlow dynamic calculation
            self.seqlen.append(seq_len)
            
            # Add a random or linear int sequence (50% prob)
            if random.random() < .5 or True:    
                
                injection_start_time_seconds = np.random.uniform(*injection_start_time_seconds_range)
                time_interval_seconds = np.random.uniform(*time_interval_seconds_range)
                time_interval_minutes = time_interval_seconds/60
                scan_time_seconds = seq_len * time_interval_seconds

                # Adjust for Unrealistically late injection time. Do this in a one liner later.
                while injection_start_time_seconds > .8*scan_time_seconds:
                    injection_start_time_seconds = np.random.uniform(*injection_start_time_seconds_range)

                AIF = parker_model_AIF(scan_time_seconds, injection_start_time_seconds, time_interval_seconds, timepoints=seq_len)

                true_ktrans = np.random.uniform(*ktrans_range)
                true_ve = np.random.uniform(*ve_range)

                ktrans = true_ktrans + np.zeros((patch_x, patch_y))
                ve = true_ve + np.zeros((patch_x, patch_y))

                Concentration = np.array(estimate_concentration_general([ktrans, ve], AIF, time_interval_minutes))

                Concentration = Concentration * np.abs(np.random.normal(1, .25, Concentration.shape))

                Intensity = revert_concentration_to_intensity(data_numpy=Concentration, reference_data_numpy=[], T1_tissue=np.random.uniform(*T1_range), TR=np.random.uniform(*TR_range), flip_angle_degrees=np.random.uniform(*flip_angle_degrees_range), injection_start_time_seconds=injection_start_time_seconds, relaxivity=np.random.uniform(*relaxivity_range), time_interval_seconds=time_interval_seconds, hematocrit=np.random.uniform(*hematocrit_range), T1_blood=0, T1_map=None, static_baseline=np.random.uniform(*baseline_intensity)).tolist()

                s = np.expand_dims(Intensity, -1)

                self.data.append(s)
                self.labels.append([true_ktrans, true_ve])

        self.batch_id = 0

def convert_v9_phantoms():

    for folder in glob.glob('../tofts_v9_phantom/QIBA_v9_Tofts/QIBA_v9_Tofts_GE_Orig/*/'):

        files = glob.glob(os.path.join(folder, 'DICOM', '*'))
        files = sorted(files)

        output_array = None

        for file in files:

            print file
            array = dicom.read_file(file).pixel_array.T[..., np.newaxis]
            print array.shape

            if output_array is None:
                output_array = array
            else:
                print output_array.shape, array.shape
                output_array = np.concatenate((output_array, array), axis=2)

        output_filepath = os.path.basename(os.path.dirname(folder)) + '.nii.gz'
        output_filepath = os.path.join('../tofts_v9_phantom/', output_filepath)
        save_numpy_2_nifti(output_array, None, output_filepath)

if __name__ == '__main__':
    convert_v9_phantoms()