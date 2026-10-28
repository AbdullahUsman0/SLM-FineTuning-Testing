import socket,sys,pytest
_original=socket.socket.connect
def guarded_connect(self,address):
 if isinstance(address,tuple) and address[0] not in ('127.0.0.1','::1','localhost'):raise RuntimeError('Outbound network disabled during offline tests')
 return _original(self,address)
socket.socket.connect=guarded_connect
raise SystemExit(pytest.main(sys.argv[1:]))
