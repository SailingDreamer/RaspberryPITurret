# import socket

# print("Creating server...")
# s = socket.socket()
# s.bind(('192.168.4.1', 10002))
# s.listen(0)

# while True:
#         client, addr = s.accept()
#         while True:
#                 content = client.recv(32)
#                 #client.send
#                 if len(content) == 0:
#                         break
#                 else:
#                         print(content)
#         print("Closing connection")
#         client.close()

import socket          
import time    

s = socket.socket()        
host = '192.168.4.1' 
port = 79               

def formatAndSend(yaw, pitch, trig):
    DTCPMessage = zeroize(yaw) + zeroize(pitch) + str(trig) + 'a'
    # print(DTCPMessage)
    s.sendall(DTCPMessage.encode())

def zeroize(val): 
    numOfZeros = 3-len(str(val))
    return numOfZeros*'0' + str(val)


s.connect((host, port))


# while True: 
    #print( "From Server: ", s.recv(32))  #This gets printed after sometime BLOCKS SCOPE
    #s.send()
    #Dimentional Turret Command Protocol

    # Pitch - 105-50 ? 99 = 50
    # Yaw - 0 - 180
    # first 3 digits are yaw, second 3 are pitch.1401051a
    #print("starting program")
    #string = input()
 
    # Output
    #print(string)
    #DTCPMessage = "0000800a"
    #DTCPMessage = "0000800a"
    # for (pos = 50; pos <= 105; pos += 1):
	# 	// in steps of 1 degree
	# 	servo1.write(pos);
	# 	delay(20);            
	# }
	# for (pos = 0; pos <= 180; pos += 1):
	# 	servo2.write(pos);
	# 	delay(20);
	# }
    # for (pos = 105; pos >= 50; pos -= 1):
	# 	servo1.write(pos);
	# 	delay(20);
	# }
    # for (pos = 180; pos >= 0; pos -= 1):
	# 	servo2.write(pos);
	# 	delay(20);
	
    # DTCPMessage = input("prompt")

# for i in range(0,10):
#     formatAndSend(180,99,1)
#     time.sleep(1)
#     formatAndSend(0,50,1)
#     time.sleep(1)


# for i in range(0, 180):
#     formatAndSend(i,50,1)
#     time.sleep(0.2)
#     # DTCPMessage = input("prompt")
#     # s.sendall(DTCPMessage.encode())
#     #time.sleep(5)
# for i in range(50, 100):
#     formatAndSend(180,i,1)
#     print(i)
#     time.sleep(0.2)
#     # DTCPMessage = input("prompt")
#     # s.sendall(DTCPMessage.encode())
#     #time.sleep(5)
# for i in range(180, 0, -1):
#     formatAndSend(i,99,1)
#     time.sleep(0.2)
#     # DTCPMessage = input("prompt")
#     # s.sendall(DTCPMessage.encode())
#     #time.sleep(5)
# for i in range(99, 50, -1):
#     formatAndSend(0,i,1)
#     print(i)
#     time.sleep(0.2)
#     # DTCPMessage = input("prompt")
#     # s.sendall(DTCPMessage.encode())
#     #time.sleep(5)

while(True):
    for i in range(0, 100):
        formatAndSend(i,60,1)
        time.sleep(0.1)
        # DTCPMessage = input("prompt")
        # s.sendall(DTCPMessage.encode())
        #time.sleep(5)
    for i in range(60, 99):#down
        formatAndSend(100,i,1)
        time.sleep(0.1)
        # DTCPMessage = input("prompt")
        # s.sendall(DTCPMessage.encode())
        #time.sleep(5)
    for i in range(100, 0, -1):
        formatAndSend(i,99,1)
        time.sleep(0.1)
        # DTCPMessage = input("prompt")
        # s.sendall(DTCPMessage.encode())
        #time.sleep(5)
    for i in range(99, 60, -1):
        formatAndSend(0,i,1)
        time.sleep(0.1)
        # DTCPMessage = input("prompt")
        # s.sendall(DTCPMessage.encode())
        #time.sleep(5)


	

s.close()               