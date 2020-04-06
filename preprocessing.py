import numpy as np
import time
import h5py

def get_patient(iPt):
    """ Converts patient index to patient number

    :param iPt: patient index (from 0 to 14)
    :return:
    """
    patient = ['23_002', '23_003', '23_004', '23_005', '23_006', '23_007', '24_001', '24_002', '24_004', '24_005',
               '25_001', '25_002', '25_003', '25_004', '25_005']
    return patient[iPt-1]


def get_record_start(iPt):
    record_start = [1276155634, 1280793545, 1289786494, 1289352195, 1304995738, 1307499627, 1279855048, 1290132962,
                    1306460580, 1307429366, 1278571538, 1278627258, 1280717229, 1290645207, 1304674450]

    return record_start[iPt-1]


def get_fs(iPt):
    fs_array = [399.6098, 399.6083, 399.6158, 399.6106, 399.6099, 399.6029, 399.6028, 399.6047, 399.6100, 399.6028,
                399.6075, 399.5972, 399.6204, 399.6132, 399.5972]
    return fs_array[iPt-1]


def get_min(file_base, t_file_start, fs, start=0, end=60):
    """Extracts one .mat file, i.e. up to one minute of data

    :param file_base: Location of data
    :param t_file_start: s, time since epoch for the start of file, should be a multiple of 60
    :param fs: Hz, sampling frequency
    :param start: s, seconds into minute chunk to start pulling data
    :param end: s, seconds into minute chunk to stop pulling data
    :return: data_segment: numpy array (16 x timesteps
    """

    utc = time.gmtime(t_file_start)  # convert to UTC date format

    filename = file_base + time.strftime("/Data_%Y_%m_%d/Hour_%H/UTC_%H_%M_00.mat", utc)  # converts UTC to filename
    # Loads .mat file into data_segment
    f = h5py.File(filename)
    for k, v in f.items(): # This assumes there is only one variable (the data matrix) in the .mat file, otherwise use commented line
        data_segment = np.array(v)  # 16 * timebins numpy array
	# arrays[k] = np.array(v) # if more than one variable in .mat file, what original code used

    #return data from 'start' to 'end' seconds added .5 as int() floors values
    return data_segment[:, int(start*fs+.5):int(end*fs+.5)]
    #return np.arange(0,160).reshape(16,10)


def get_data(iPt, t_start, t_end):
    """Collects and concatenates required data segment from saved .mat files

    :param iPt: patient index (1 to 15)
    :param t_start: s, seconds since start of recording, start of data to be extracted
    :param t_end: s, seconds since start of recording, end of data to be extracted
    :return: data: numpy matrix (16 x timesteps)
    """


    if t_end <= t_start:
        print('Error: end time not after start time')
        exit()

    # NOTE: start times are generated from date strings assuming times are gm time. While this is wrong, if we always assume gm this should be fine
    # EXCEPT: day light savings time throws it out as it would occur a tthe wrong time!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # s, Time of EEG recording start since epoch (1/1/1970)

    # code to extract these values from string:
    #   calendar.timegm(time.strptime('06-May-2011 09:34:10', '%d-%b-20%y %H:%M:%S'))

    # location of data
    file_base = '/media/NVdata/Patient_' + get_patient(iPt)

    # iEEG frequencies, for each patient
    fs = get_fs(iPt)

    # Convert times from time since start of recording to time since epoch
    t_start = t_start + get_record_start(iPt)
    t_end = t_end + get_record_start(iPt)

    # Convert times to UTC date format
    UTC_start = time.gmtime(t_start)
    UTC_end = time.gmtime(t_end)

    # if start and end time in the same .mat file
    if time.gmtime(t_start-t_start % 60) == time.gmtime(t_end-t_end % 60):

        data = get_min(file_base, t_start-t_start % 60, fs, UTC_start.tm_sec, UTC_end.tm_sec)   # gets data from the file

    else: # extract from multiple files
        # extract appropriate segment of the first file
        data = get_min(file_base, t_start - t_start % 60, fs, UTC_start.tm_sec, 60)
        t_tmp = t_start - t_start % 60 + 60  # moves to next time block

        # while not at the start of the last file to look at
        while t_tmp < (t_end - t_end % 60):
            # extract full file and join to previous data
            data = np.concatenate((data, get_min(file_base, t_tmp, fs)), axis=1)
            t_tmp += 60 # advance to next segment

        # extract last segment of file to the appropriate point and join to previous data
        data = np.concatenate((data, get_min(file_base, t_tmp, fs, 0, UTC_end.tm_sec)), axis=1)

    return data


