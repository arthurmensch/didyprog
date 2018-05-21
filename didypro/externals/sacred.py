import os
from os.path import join

from sacred.observers import MongoObserver, FileStorageObserver


def get_artifact_dir(run):
    """Get artifact directory for a run observed by a FileStorageObserver"""
    artifact_dir = join(run.observers[0].basedir, str(run._id), 'artifacts')
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)
    return artifact_dir


def lazy_add_artifact(run, name, filename):
    """Record the artifact filename within the record of the current run"""
    if not run.unobserved:
        for observer in run.observers:
            if isinstance(observer, MongoObserver):
                observer.run_entry['artifacts'].append({'name': name,
                                                        'file_id': 0,
                                                        'filename': filename})
                observer.save()
            elif isinstance(observer, FileStorageObserver):
                observer.run_entry['artifacts'].append(name)
                observer.save_json(observer.run_entry, 'run.json')