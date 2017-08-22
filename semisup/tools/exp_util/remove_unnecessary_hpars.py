import sys
sys.path.append('../../..')
from semisup.tools.data_dirs import tinydb_path
import os
os.chdir('../../../')
from tinydb import TinyDB, Query
from shutil import copy

# copy for backup
print 'doing backup of result json file...'
copy(tinydb_path, tinydb_path[:-5] + '(backup).json')


db = TinyDB(tinydb_path)
hptb = db.table('lba-hyper-params')
hp = Query()

is_empty = lambda s: not os.path.exists(s)

print 'found %d empty hyper params. deleting...' %\
      len(hptb.search(hp.logdir.test(is_empty)))

hptb.remove(hp.logdir.test(is_empty))

print 'done.'