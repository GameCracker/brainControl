from open_bci import *
from multiprocessing import process
from boto.s3.connection import S3Connection
from boto.s3.key import Key
import boto.s3.connection
import csv
import time, datetime
import threading
import boto.s3
import sys, os
import redis, inspect
from inspect import currentframe, getframeinfo

class DataTransfer(object):

	# def __init__(self, filename):
	def __init__(self):
		filename = ""
		self.file_name = filename
		self.local_path = "/Users/ziqipeng/Dropbox/bci/system_zero/data/motor_imagery"
		self.local_data = self.local_path + filename
		self.aws_access_key_id = ''
		self.aws_secret_access_key = ''
		self.bucket_name = self.aws_access_key_id.lower() + '-eegdata'
		self.exist_bucket = 'kellystorage0'
		self.filename = "/Users/ziqipeng/Dropbox/bci/system_zero/data/motor_imagery/motor_imagery_1.csv"
		self.folder = "/motor_imager/"
		self.redis_name = ""
		self.redis_path = "localhost"

	def uploader(self, t, obj, content):
		conn = boto.connect_s3(aws_access_key_id=self.aws_access_key_id,
			aws_secret_access_key=self.aws_secret_access_key,
			calling_format=boto.s3.connection.OrdinaryCallingFormat(),)
		# redis for t = 0
		bucket = conn.create_bucket(self.bucket_name)
		if t != 0:
			object_name = self.filename.split('/')[-1]
			key = bucket.new_key(self.folder_name + object_name)
			key.set_contents_from_filename(self.filename)
		else:
			key = bucket.new_key(obj)
			key.set_contents_from_string(content)
		return 1

	def downloader(self):
		print ""

	# print frameinfo.filename, frameinfo.lineno
	def file_info(self):
		frameinfo = getframeinfo(currentframe())

	def transfer(self):
		r_server = redis.Redis(self.redis_path)
		# r_server.set("name", "Hello Kitty")
		# test = r_server.get("name")
		# print "line %d test=%s" %(self.lineno(), test)
		r_server.sadd("test:0000:attr", "userid:0002")
		# tmp = r_server.smembers(self.redis_name)
		tmp = r_server.smembers("test:0000:attr")
		print "line %d tmp=%s, type(tmp)=%s" %(self.lineno(), str(tmp), type(tmp))
		# if self.uploader(0, self.redis_name, str(tmp)) == 1:
		if self.uploader(0, "test:0000:attr", str(tmp)) == 1:
			# datetime.datetime.now()
			r_server.delete("members")
			# r_server.delete(self.redis_name)
		print "line %d" %self.lineno()
		
	def lineno(self):
		return inspect.currentframe().f_back.f_lineno

		# bucket = conn.get_bucket(self.exist_bucket, validate=False)

		# key = bucket.new_key('motor_imagery/motor_imagery_1.csv')
		# key.set_contents_from_filename(self.filename)
		# key.get_contents_as_string('test 0')

		# key.set_acl('public-read')

		# k = Key(bucket)
		# k.key = 'motor_imagery_1'
		# k.get_contents_as_string('test 0')
		# k.set_contents_from_filename(self.filename)
		# print "{name}".format(name=bucket.name)
		# for bucket in conn.get_all_buckets():
		# 	print "{name}\t{created}".format(name=bucket_name, created=bucket.created_date,)
		# print "conn %s" %conn
		# bucket = conn.lookup('donebox-static')
		# key = bucket.new_key('textkey')
		# key.set_contents_from_filename('This is a test')
		# key.delete()
		# bucket = conn.create_bucket(self.bucket_name)
		# # bucket = conn.create_bucket(self.bucket_name, location=boto.s3.connection.Location.DEFAULT)
		# print "Uploading %s to Amazon S3 bucket %s" % (self.filename, self.bucket_name)
		# from boto.s3.key import key
		# k = Key(bucket)
		# k.key = self.filename
		# k.set_contents_from_filename(self.filename, cb=percent_cb, num_cb=10)

	def percent_cb(self, complete, total):
		sys.stdout.write('.')
		sys.stdout.flush()

	def csv_data_loader(self):
		dirc = self.local_data
		n = 0
		df = pd.read_csv(dirc)
		print df.shape
		w = df.shape[0]
		l = df.shape[1]
		n = 0
		t = []
		t0 = 0.0
		for row in df.iterrows():
			if n >= 200:
				r = row[1].tolist()
				c = 1
				for i in range(8):
					if (r[i] != r[i]) or (r[i] == None) or (len(r) != 10):
						c = 0
						break
				if c == 1:
					if 'X' in locals():
						X = np.vstack([X, r[:8]])
						t.append(r[8] - t0)
					else:
						X = r[:8]
						t0 = r[8]
						t.append(0.0)
						print type(t0)
			n += 1
		return X, t

if __name__ == "__main__":
	dt = DataTransfer()
	dt.transfer()

