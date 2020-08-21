# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 02:06:25 2019

@author: mnagd
"""

import socket
sock =  socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('192.168.0.133', 10004))

while True:
    data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
    print ("received message:", data)