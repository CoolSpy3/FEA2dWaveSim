from abc import ABC, abstractmethod
import math

class Geometry(ABC):
	def __init__(self, use_exact_coordinates=False):
		self.use_exact_coordinates = use_exact_coordinates

	@abstractmethod
	def contains_point(self, x, y):
		...

	def contains_raw_point(self, x, y, x_step, y_step):
		return \
			self.contains_point(x, y) if self.use_exact_coordinates \
				else self.contains_point(x * x_step, y * y_step)

class Point(Geometry):
	def __init__(self, x, y, rtol=0.01, atol=0, use_exact_coordinates=False):
		super().__init__(use_exact_coordinates)
		self.x, self.y, self.rtol, self.atol = x, y, rtol, atol

	def contains_point(self, x, y):
		return math.isclose(x, self.x, rel_tol=self.rtol, abs_tol=self.atol) and \
			math.isclose(y, self.y, rel_tol=self.rtol, abs_tol=self.atol)

class Circle(Geometry):
	def __init__(self, x, y, r, use_exact_coordinates=False):
		super().__init__(use_exact_coordinates)
		self.c, self.r = (x, y), r

	def dist_from_center(self, x, y):
		return math.dist((x, y), self.c)

	def contains_point(self, x, y):
		return self.dist_from_center(x, y) <= self.r

class Rectangle(Geometry):
	def __init__(self, x, y, width, height, use_exact_coordinates=False):
		super().__init__(use_exact_coordinates)
		# When everything is discretized, distances are one less because the first square has length 1
		# In the infinitesimal case, we're still off by one element of the step size,
		# but I can't think of a way around this that's not more trouble than its worth for a (hopefully) small correction
		# The weird copy-sign stuff is just to support negative dimensions (because why not)
		if use_exact_coordinates:
			width = math.copysign(abs(width)-1, width)
			height = math.copysign(abs(height)-1, height)

		x1, x2, y1, y2 = x, x + width, y, y + height
		self.x_min, self.x_max = min(x1, x2), max(x1, x2)
		self.y_min, self.y_max = min(y1, y2), max(y1, y2)

	def dist_to_border(self, x, y):
		p = x, y
		if self.contains_point(x, y):
			is_left = x < (self.x_max + self.x_min) / 2
			is_down = y < (self.y_max + self.y_min) / 2
			h_dist = abs(x - (self.x_min if is_left else self.x_max))
			v_dist = abs(y - (self.y_min if is_down else self.y_max))
			return min(h_dist, v_dist)
		else:
			closest_point = (
				self.x_min if x < self.x_min else self.x_max if x > self.x_max else x
				self.y_min if y < self.y_min else self.y_max if y > self.y_max else y
			)
			return math.dist(p, closest_point)

	def contains_point(self, x, y):
		return self.x_min <= x <= self.x_max and \
			self.y_min <= y <= self.y_max

class Border(Geometry):
	def __init__(self, x, y, width, height, thickness, inside_border=True, use_exact_coordinates=False):
		super().__init__(True)  # Pass coordinate conversions to the rectangles
		self.thickness, self.inside_border = thickness, inside_border
		if inside_border:
			thickness *= -1
		rect1 = Rectangle(x, y, width, height, use_exact_coordinates)
		rect2 = Rectangle(
			rect1.x_min - thickness,
			rect1.y_min - thickness,
			abs(width) + 2 * thickness,
			abs(height) + 2 * thickness,
			use_exact_coordinates
		)
		self.inner_rect, self.outer_rect = \
			(rect2, rect1) if inside_border else (rect1, rect2)

	def contains_point(self, x, y):
		return self.outer_rect.contains_point(x, y) and not \
			self.inner_rect.contains_point(x, y)

	def contains_raw_point(self, x, y, x_step, y_step):
		return self.outer_rect.contains_raw_point(x, y, x_step, y_step) and not \
			self.inner_rect.contains_raw_point(x, y, x_step, y_step)
