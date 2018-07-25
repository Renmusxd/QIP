import json


class StateSetup:
    def __init__(self, initialstates, n):
        self.initialstates = initialstates
        self.n = n

    def to_json(self):
        return json.dumps({
            'command': 'StateSetup',
            'initialstates': self.initialstates,
            'n': self.n
        })

    @staticmethod
    def from_json(msg):
        json_dict = json.loads(msg)
        if json_dict['command'] == 'StateSetup':
            return StateSetup(json_dict['initialstates'], json_dict['n'])
        else:
            raise ValueError('Wrong command for StateSetup')


class WorkerSetup:
    def __init__(self, initialstates, indexstart, indexend):
        self.initialstates = initialstates
        self.indexstart = indexstart
        self.indexend = indexend

    def to_json(self):
        return json.dumps({
             'command': 'WorkerSetup',
             'initialstates': self.initialstates,
             'indexstart': self.indexstart,
             'indexend': self.indexend
        })

    @staticmethod
    def from_json(msg):
        json_dict = json.loads(msg)
        if json_dict['command'] == 'WorkerSetup':
            return WorkerSetup(json_dict['initialstates'], json_dict['indexstart'], json_dict['indexend'])
        else:
            raise ValueError('Wrong command for WorkerSetup')


class WorkerOperation:
    DONE = 'done'
    KRONPROD = 'kronprod'
    MEASURE = 'measure'

    def __init__(self, opcommand, job_id, **kwargs):
        # TODO add various operations
        self.opcommand = opcommand
        self.job_id = job_id
        self.kwargs = kwargs

    def to_json(self):
        return json.dumps(dict(command='WorkerOperation', opcommand=self.opcommand, job_id=self.job_id,
                               **self.kwargs))

    @staticmethod
    def from_json(msg):
        json_dict = json.loads(msg)
        if json_dict['command'] == 'WorkerOperation':
            return WorkerOperation(**json_dict)
        else:
            raise ValueError('Wrong command for WorkerOperation')


class WorkerDone:
    def __init__(self, job_id):
        self.job_id = job_id

    def to_json(self):
        return json.dumps({'command': 'WorkerDone',
                           'job_id': self.job_id})

    @staticmethod
    def from_json(msg):
        json_dict = json.loads(msg)
        if json_dict['command'] == 'WorkerDone':
            return WorkerDone(json_dict['job_id'])
        else:
            raise ValueError('Wrong command for WorkerDone')


class WorkerSyncCommand:
    def __init__(self, job_id):
        self.job_id = job_id

    def to_json(self):
        return json.dumps({'command': 'WorkerSyncCommand',
                           'job_id': self.job_id})

    @staticmethod
    def from_json(msg):
        json_dict = json.loads(msg)
        if json_dict['command'] == 'WorkerSyncCommand':
            return WorkerSyncCommand(json_dict['job_id'])
        else:
            raise ValueError('Wrong command for WorkerSyncCommand')
