GymGuard Server (RabbitMQ)


Prerequisite:
1. /models directory needs keras model and label mapping text as generated using GymGuard Client Trainer
2. RabbitMQ/Erlang installed on system
RabbitMQ Docs for Installation: https://www.rabbitmq.com/docs/download


To run server (listen):

1. Run `python main.py`

## To use this, you need to send RGB frames. The GymGuard client (windows exec) can be used.
Use GymGyuard Client in the path : `/client/app.py`