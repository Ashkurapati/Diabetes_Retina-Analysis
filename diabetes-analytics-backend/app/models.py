from sqlalchemy import Column, Integer, String, Float
from .db import Base

class RetinaLabel(Base):
    __tablename__ = "retina_labels"
    image_id = Column(String(64), primary_key=True)
    patient_id = Column(Integer)
    patient_age = Column(Integer)
    diabetes_time_y = Column(Float)
    # ...add all other columns you have...
    DR_ICDR = Column(Integer)
