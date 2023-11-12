import logging
import os
import uuid
from datetime import datetime
from typing import Annotated

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.params import Header
from sqlalchemy.orm import Session

from api.cvision import analyze_video
from count_frames import count_frames_and_duration
from database import engine
from models.records import Record

app = FastAPI(
    title='Вижн — Leaders hack',
    description='API',
    docs_url='/docs'
)


# @app.post('/signup', summary="Create new user", response_model=TokenSchema)
# async def create_user(data: UserAuth, db: Session = Depends(get_db)):
#     # querying database to check if user already exist
#     user = db.query(User).filter(User.email == data.email).first()
#     if user is not None:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="User with this email already exist"
#         )
#     user = crud.create_user(db, data.username, data.email, data.password, data.first_name, data.last_name)
#     return get_login_response(user)
#
#
# @app.post('/login', summary="Create access and refresh tokens for user by email (not username!)",
#           response_model=TokenSchema)
# async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
#     user = crud.get_user_by_email(db, form_data.username)
#     if user is None:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="Incorrect email or password"
#         )
#     hashed_pass = user.password
#     if not verify_password(form_data.password, hashed_pass):
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="Incorrect email or password"
#         )
#
#     return get_login_response(user)
#
#
# @app.get("/api/me", tags=["users"])
# async def get_me(user: User = Depends(get_current_user)):
#     return user.to_dict(rules=['-certificates.user'])


@app.post('/upload_video')
async def upload_video(file: Annotated[UploadFile, File(description="A file read as UploadFile")], date: datetime,
                       location: str, user_agent: Annotated[str | None, Header()] = None):
    print(file.filename)
    file_type, file_extens = file.content_type.split('/')
    if file_type != 'video':
        raise HTTPException(status_code=415, detail=f'Wrong filetype. Expected video, got {file_type}')
    uuid_ = str(uuid.uuid4())
    os.makedirs(f'{os.path.realpath(__file__).replace("/main.py", "")}/data_storage/videos/{uuid_}')
    new_path = f'data_storage/videos/{uuid_}/{file.filename}'
    with open(new_path, 'wb') as f:
        f.write(file.file.read())
    num_of_frames, duration = count_frames_and_duration(new_path)
    result = analyze_video(new_path)
    with Session(engine) as session:
        record = Record(
            date=date,
            location=location,
            uuid=uuid_,
            file_path=new_path,
            request_headers=user_agent,
            processing_time=0,
            video_len=duration,
            classes=result
        )
        session.add(record)
        session.commit()
        return record.to_dict()


@app.get('/get_stats')
async def get_stats(location: str = None, date_start: datetime = None, date_end: datetime = None):
    with Session(engine) as session:
        if location is None:
            records = session.query(Record).filter().all()
        elif date_start is not None and date_end is not None:
            records = session.query(Record).filter(Record.location == location, Record.date >= date_start,
                                                   Record.date <= date_end).all()
        elif date_start is None or date_end is None:
            records = session.query(Record).filter(Record.location == location).all()
        else:
            records = session.query(Record).filter(Record.location == location, Record.date >= date_start,
                                                   Record.date <= date_end).all()
        return {'records': [record.to_dict() for record in records]}
