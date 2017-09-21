#!/usr/bin/env python
import collections

''' This is a configuration template for the selector class.'''

selector_config_template = collections.namedtuple('selector_config_template',

[

    # This work area will be created and the entire output will appear there
    'work_area',

    # job type: Not really used for now in the code
    'job_type',

    # takes the csv file, which includes the info about the CRF and image locations and a set of pictures to be used in the algo
    'input_csv',

    # Time interval in units of days. The reference point is the time from the CRF form. TO BE FINALIZED ONCE CRF IS IN PLACE
    'time_interval',

    # selection percentage for train/test ratio
    'split_train_test',

    # random_state for train/test ratio
    'split_random_seed',

    # Train/test selection based on scans. Two methods are availabe: 1. split the dataset in burnarea  unites (all scans taking in differnet time will be treated as a unit); 2. split the dataset in session (the unit of the dataset is a scan regardless when it was taken)
    'split_method', # 'burnarea' # or session

    # Enable/disable logs
    'keep_logs', # not clear yet how to use it

    # How many pics user wants to provide (there are 407 in total) - > this file will apply to all datasets. If you leave blanck this line, then the file from the csv file will be applied to each data set
    'gpic_selection_file'

]
 
                                        )
# create an instance of the namedtuple template
selector_config = selector_config_template('', '','',0,0,0,'',False,'')


### cross-validation part

selector_cross_val_config_template = collections.namedtuple('selector_cross_val_config_template',

[

        # This work area will be created and the entire output will appear there
        'work_area',

        # job type: Not really used for now in the code
        'job_type',

        # takes the csv file, which includes the info about the CRF and image locations and a set of pictures to be used in the algo
        'input_csv',

        # Time interval in units of days. The reference point is the time from the CRF form. TO BE FINALIZED ONCE CRF IS IN PLACE
        'time_interval',

        # selection percentage for train/test ratio
        'n_split',

        # random_state for train/test ratio
        'shuffle',

        'random_state',

        # Train/test selection based on scans. Two methods are availabe: 1. split the dataset in burnarea  unites (all scans taking in differnet time will be treated as a unit); 2. split the dataset in session (the unit of the dataset is a scan regardless when it was taken)
        'split_method', # 'burnarea' # or session

        # Enable/disable logs
        'keep_logs', # not clear yet how to use it

        # How many pics user wants to provide (there are 407 in total) - > this file will apply to all datasets. If you leave blanck this line, then the file from the csv file will be applied to each data set
        'gpic_selection_file'
]
)
# create an instance of the namedtuple template
selector_cross_val_config = selector_cross_val_config_template([''], '','',0,0,False, None, '',False,'')

                                         





# Brian's request

#pretestOp
#pretestArg

#pretrainOp
#pretrainArg

