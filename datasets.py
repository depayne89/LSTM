import h5py
import numpy as np
from sys import stdout
from bisect import bisect
import time


def pt_list():
    return np.array([1,6,8,9,10,11,13,15])


def get_patient(iPt):
    """ Converts patient index to patient number

    :param iPt: patient index (from 1 to 15)
    :return:
    """
    patient = ['23_002', '23_003', '23_004', '23_005', '23_006', '23_007', '24_001', '24_002', '24_004', '24_005',
               '25_001', '25_002', '25_003', '25_004', '25_005']
    return patient[iPt-1]


def get_annots(iPt):
    """Retrieves annotations to the NV dataset for one patient

    :param iPt: patient number (from 1 to 15)
    :return:
    SzDur: s, seizure duration
    SzInd: index
    SzTimes: microseconds, start time of seizure, relative to recording start
    SzType: 1,2 or 3, seizure type according to NeuroVista classification
    """

    # File location
    filename = '/media/NVdata/Annotations/' + get_patient(iPt) + '_Annots.mat'

    # Loads .mat file
    f = h5py.File(filename, 'r')

    # Extract each variable in the mat file
    arrays = {}
    for k, v in f.items():
        arrays[k] = np.array(v) # if more than one variable in .mat file, what original code used

    return arrays['SzDur'], arrays['SzInd'], arrays['SzTimes'], arrays['SzType']


def get_record_start(iPt):
    record_start = [1276155634, 1280793545, 1289786494, 1289352195, 1304995738, 1307499627, 1279855048, 1290132962,
                    1306460580, 1307429366, 1278571538, 1278627258, 1280717229, 1290645207, 1304674450]
    return record_start[iPt-1]


def get_record_length(iPt):
    lengths = [64044366.,  63081236.1, 39613596.,  20129350.5, 23574790.9, 38125136.1,
     15973014.8, 48249270.7, 34121630.8, 32244980.8, 62349005.,  62983441.4,
     60782771.,  54173421.9, 40232047.2]
    return lengths[iPt-1]


def select_seizures(sz_times, sz_type, latest_time, earliest_day = 100, lead_time = 0):
    ''' Selects seizure times for type 1 or 2 lead seizures after day 100

    :param SzTimes: s, array of seizure start times, from time of recording start
    :param SzType: 1, 2 or 3, denoting seizure types
    :param earliest_day: days, first day included in the experiment, default = 100
    :return: s, array of valid seizure times
    '''

    #  Remove type 3
    SzTimes_not3 = sz_times[sz_type != 3]

    # lead seizures only (5+ hrs since last seizure)
    ISI = np.append(float('inf'),
                    (SzTimes_not3[1:] - SzTimes_not3[:-1]) / 60 / 60)  # hr, Calculate inter seizure interval
    SzTimes_n3_lead = SzTimes_not3[ISI > lead_time]  # Lead seizures only

    # Remove seizures in first 100 days (and days from end if invalid)

    SzTimes_n3_ld_100 = SzTimes_n3_lead[np.logical_and(SzTimes_n3_lead / 3600 / 24 > earliest_day,SzTimes_n3_lead < latest_time)]  # Ignore first 100 days (and bad data at end for 1, 3, 13

    return SzTimes_n3_ld_100


def seizure_train_test_split(patient, percent_train):
    ''' Determines the times of seizures for train, test (and validaiton) groups.
    Only type 1 and 2 lead seizures after day 100 are valid

    :param patient: patient number (1-15)
    :param percent_train: 0-100, percentage of dataset in the training group
    :return:
    '''

    # stdout.write('\r Grouping seizures into train/test sets...')
    # stdout.flush()

    # get valid sz times
    [SzDur, SzInd, SzTimes, SzType] = get_annots(patient)
    SzTimes /= 1000000

    rec_length = get_record_length(patient)

    SzTimes_select = select_seizures(SzTimes, SzType, rec_length, lead_time=0)
    # Select train of seizures
    train_cutoff = np.percentile(SzTimes_select, percent_train)  # Change to 100 to just get all times

    SzTimes_train = SzTimes_select[SzTimes_select < train_cutoff]
    SzTimes_test = SzTimes_select[SzTimes_select >= train_cutoff]

    # print 'Data not saved'
    cutoff = (SzTimes_test[0] + SzTimes_train[-1])/2


    f = h5py.File('/media/NVdata/SzTimes/{0:0.0f}_{1:0.0f}_{2:0.0f}.mat'.format(\
        percent_train, (100-percent_train), patient), 'w')
    f.create_dataset('train', data=SzTimes_train)
    f.create_dataset('test', data=SzTimes_test)
    f.create_dataset('cutoff', data=cutoff)
    f.close()

    stdout.write('Complete')
    # seizure_train_test_split(iPt, 80) was run on 6/4/20

    return


def load_sz_times(patient, percent_train=80):
    ''' Loads seizure times from file

    :param patient: patient number (1 to 15)
    :param percent_train: percentage of data used for training set
    :param validation: boolean, is validation used
    :return:
    Sz_train: s, time since recording for training set seizures
    Sz_test: s, time since recording for test set seizures
    cutoff: s, end of the training set period
    '''

    f = h5py.File('/media/NVdata/SzTimes/{0:0.0f}_{1:0.0f}_{2:0.0f}.mat'.format( \
        percent_train, (100 - percent_train), patient), 'r')
    Sz_train = np.array(f['train'])
    Sz_test = np.array(f['test'])
    cutoff = f['cutoff']
    f.close()
    return Sz_train, Sz_test, cutoff


def generate_dataset(patient, percent_train, steps_back=2, train=True):
    ''' Segments train set into 10 minutes samples and lables into ictal or interictal

    :param patient: (1-15)
    :param steps_back: how many 10 minute samples before seizure to lable ictal (ie prediciton horizon = stepsback*10)
    :param train_percent: percent of seizures used in train set (default 80)
    :return:
    '''

    stdout.write('\r Generating dataset...')
    stdout.flush()

    # get end of train period
    f = h5py.File('/media/NVdata/SzTimes/{0:0.0f}_{1:0.0f}_{2:0.0f}.mat'.format( \
        percent_train, (100 - percent_train), patient), 'r')
    if train:
        start = 100*24*3600
        end = np.asarray(f['cutoff'])
        SzTimes = np.asarray(f['train'])
    else:
        start = np.asarray(f['cutoff'])
        end = get_record_length(patient)
        SzTimes = np.asarray(f['test'])

    f.close()
    print(end)
    # create array of timestamps every 10 minutes from start of period to end
    start = start + get_record_start(patient)
    start_round = start + (600-start%600)  # round up to nearest 10min mark
    start_time = start_round - get_record_start(patient)


    sample_times= np.array([])
    print('Startend ', start_time, ' ', end)

    while (start_time + 60*10) < end:
        sample_times = np.append(sample_times, start_time)
        start_time += 60*10

    labels = np.zeros(sample_times.shape)
    print(sample_times)

    for time in SzTimes:
        # print('\nSz: ', time)
        first_sample_after_sz = bisect(sample_times, time)
        for i in range(0,2):  # Sz and one after labeled with 2
            if first_sample_after_sz < sample_times.shape[0]:
                # print(' sample ', sample_times[first_sample_after_sz-i])
                labels[first_sample_after_sz-i] = 2
            # ----------- add labels 2 and -1 --------------
        for i in range(2,steps_back+2): #2 preceeding labled with 1
            # print(' sample ', sample_times[first_sample_after_sz - i])
            labels[first_sample_after_sz - i] = 1

    print('Not yet labelled dropout')

    # Save to file
    if train:
        f = h5py.File('/media/NVdata/SzTimes/all_train_%dstep_%d_%d.mat' % (steps_back, percent_train, patient), 'w')
        f.create_dataset('train_start', data = start_round)
        f.create_dataset('train_end', data = end)
        f.create_dataset('pred_horizon', data = steps_back * 10)
        f.create_dataset('sample_times', data = sample_times)
        f.create_dataset('sample_labels', data = labels)
        f.close()
    else:
        f = h5py.File('/media/NVdata/SzTimes/all_test_%dstep_%d_%d.mat' % (steps_back, percent_train, patient), 'w')
        f.create_dataset('test_start', data=start_round)
        f.create_dataset('test_end', data=end)
        f.create_dataset('pred_horizon', data=steps_back * 10)
        f.create_dataset('sample_times', data=sample_times)
        f.create_dataset('sample_labels', data=labels)
        f.close()

    stdout.write('Complete')

    return


# for iPt in pt_list():
#     print(iPt)
#     generate_dataset(iPt,80)
#     generate_dataset(iPt,80, train=False)
#     print()
