# import socket               # Import socket module

# s = socket.socket()         # Create a socket object
# host = '172.16.28.222' # Get local machine name
# port = 12345                # Reserve a port for your service.
# s.bind((host, port))        # Bind to the port
# s.listen(5)                 # Now wait for client connection.
# while True:
#    c, addr = s.accept()     # Establish connection with client.
#    print ('Got connection from', addr)
#    c.send(b'0')
#    c.close()  
#                  # Close the connection

Y_pred=1
import socket
# s = socket.socket()         # Create a socket object
host = '172.16.29.251' # Get local machine name
#host=socket.gethostname()
port = 12346            # Reserve a port for your service.
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((host, port))
Y_pred=1
s.sendall(b'%d' %Y_pred)
data = s.recv(1024)
s.close()
print('rec', repr(data))
