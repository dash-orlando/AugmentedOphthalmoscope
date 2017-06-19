from    usbProtocol                 import  createUSBPort   # Create USB Port
from    threading                   import  Thread          # Used to thread processes
from    Queue                       import  LifoQueue       # Used to queue input/output
from    time                        import  sleep, time, clock

# number of samples
n = 2500

# Initialize ToF sensor
deviceName, port, baudRate = "VL6180", 0, 115200
ToF = createUSBPort( deviceName, port, baudRate, 3 )
if ToF.is_open == False:
    ToF.open()
ToF.close()

print('Benchmark: START')
print('Sampling %i data points' %n)
print('=====================================\n')

if ToF.is_open == False:
    ToF.open()

now = clock()
dist1 = []
for i in range(1, n):
    dist1.append(int( ( ToF.read(size=2).strip('\0') ).strip('\n') ))
elapsed = ( clock()-now )

print( 'PySerial: %f' %elapsed )
print( 'Length  : %i' %len(dist1) )
#print( 'Type    : %s' %type(dist))
#print( 'Value   : %i' %dist )
ToF.close()

print('\n=====================================\n')

if ToF.is_open == False:
    ToF.open()

now = clock()
dist2 = []
for i in range(1, n):
    dist2.append( int( (ToF.readline()[:-1]).strip('\0') ) )
elapsed = ( clock()-now )

print( 'I/O     : %f' %elapsed )
print( 'Length  : %i' %len(dist2) )
#print( 'Type    : %s' %type(dist))
#print( 'Value   : %i' %dist )
ToF.close()

print('\n=====================================')
print('\nBenchmark: END')

