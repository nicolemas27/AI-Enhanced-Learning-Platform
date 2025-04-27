from db import db

def verify_data_integrity():
    """Run this periodically (e.g. daily cron job)"""
    for user in db.users.find():
        videos = db.get_video_progress(user['_id'])
        for video in videos:
            assert len(video['attempts']) > 0
            assert 'aggregates' in video

#from cryptography.fernet import Fernet
#print(Fernet.generate_key().decode())