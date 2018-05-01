import socket               # Import socket module

s = socket.socket()         # Create a socket object
host = '172.16.28.222' # Get local machine name
# host=socket.gethostname()
port = 12345                # Reserve a port for your service.
# s = socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((host, port))
s.sendall(b'0')
data = s.recv(1024)
s.close()
print('rec', repr(data))