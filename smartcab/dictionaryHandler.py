class Dict(dict):
	def __add__(self, key):
		self[key] = key