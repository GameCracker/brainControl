from open_bci import *
import time, csv, redis, inspect, serial

class OpenbciCollect(object):

	def __init__(self):
		self.obci = OpenBCIBoard()

	def to_database(self):
		print ""

	def display_data(self, data):
		print data

	def receive_data(self):
		self.obci.start(self.display_data)

if __name__ == "__main__":
	# ser = serial.Serial(0)
	# print ser.name
	# serial = serial.Serial('/dev/tty.usbserial', 9600)
	# while True:
	# 	print ser.readline()
	oc = OpenbciCollect()
	oc.receive_data()



