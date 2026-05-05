from abc import ABC, abstractmethod
import math

class Geometry(ABC):
	"""A base class for all geometric objects"""
	def __init__(self, use_exact_coordinates=False):
		"""
		Set use_exact_coordinates to True if you want to define
		things in terms of grid coordinates
		Otherwise, this will handle scaling so that
		changes in the simulation resolution don't move this Geometry
		"""
		self.use_exact_coordinates = use_exact_coordinates

	@abstractmethod
	def _contains_point(self, x, y):
		"""
		To be implemented by subclasses.
		Return True if the given point lies within the Geometry.
		This should only be called by Geometry classes. Otherwise,
		we can't be sure of unit conversions.
		"""
		...

	def contains_raw_point(self, x, y, x_step, y_step):
		"""
		Converts x and y into real coordinates (if necessary)
		and calls _contains_point
		"""
		return \
			self._contains_point(x, y) if self.use_exact_coordinates \
				else self._contains_point(x * x_step, y * y_step)

# The rest of this is just defining some geometric primitives that we can use

class Circle(Geometry):
	def __init__(self, x, y, r, use_exact_coordinates=False):
		super().__init__(use_exact_coordinates)
		self.c, self.r = (x, y), r

	def dist_from_center(self, x, y):
		"""Returns the distance of a point from the center"""
		return math.dist((x, y), self.c)

	def _contains_point(self, x, y):
		return self.dist_from_center(x, y) <= self.r

class Point(Circle):
	"""
	Points should include stuff around them to account for floating-point weirdness.
	This makes them the same as small circles!
	"""
	def __init__(self, x, y, use_exact_coordinates=False, default_radius=1):
		"""
		There should be no need to change default_radius, it should be automatically updated as-appropriate
		If you're doing something weird though, you now have the power!
		"""
		super().__init__(x, y, default_radius, use_exact_coordinates)

	def contains_raw_point(self, x, y, x_step, y_step):
		# If using exact coordinates, everything should be integers, so permit no deviation!
		# Otherwise, we may deviate no more than one step from the desired point.
		self.r = 0 if self.use_exact_coordinates else max(x_step, y_step)
		return super().contains_raw_point(x, y, x_step, y_step)

class Rectangle(Geometry):
	def __init__(self, x, y, width, height, use_exact_coordinates=False):
		super().__init__(use_exact_coordinates)
		# When everything is discretized, distances are one less because the first square has length 1
		# In the infinitesimal case, we're still off by one element of the step size,
		# but I can't think of a way around this that's not more trouble than its worth for a (hopefully) small correction
		# The weird copy-sign stuff is just to support negative dimensions (because why not)
		# For a fine enough step size, this makes everything work basically as one would expect
		if use_exact_coordinates:
			width = math.copysign(abs(width)-1, width)
			height = math.copysign(abs(height)-1, height)

		x1, x2, y1, y2 = x, x + width, y, y + height
		self.x_min, self.x_max = min(x1, x2), max(x1, x2)
		self.y_min, self.y_max = min(y1, y2), max(y1, y2)

	def dist_to_border(self, x, y):
		"""Returns the distance of a point to the edge of the rectangle"""
		p = x, y
		# We need to use different logic depending on if we're inside or outside the rectangle
		if self._contains_point(x, y):
			# If we're inside, find which two edges the point is closer to
			# by checking whether x and y are below their respective midpoints
			is_left = x < (self.x_max + self.x_min) / 2
			is_down = y < (self.y_max + self.y_min) / 2
			# Then find the distance along each of those coordinates to an edge
			h_dist = abs(x - (self.x_min if is_left else self.x_max))
			v_dist = abs(y - (self.y_min if is_down else self.y_max))
			# Then, take whichever one's lower
			return min(h_dist, v_dist)
		else:
			# If we're outside the rectangle, find the closest point on an edge.
			# If we're in the bounds on one coordinate, the closest distance is a
			# perpendicular line that leaves that coordinate constant, so the closest
			# point has the same coordinate value as whatever we were passed.
			# Otherwise, its one of the extremes depending on which way it overshoots
			closest_point = (
				self.x_min if x < self.x_min else self.x_max if x > self.x_max else x,
				self.y_min if y < self.y_min else self.y_max if y > self.y_max else y
			)
			# Once we've found the closest point, just take the distance to it
			return math.dist(p, closest_point)

	def _contains_point(self, x, y):
		return self.x_min <= x <= self.x_max and \
			self.y_min <= y <= self.y_max

class Border(Geometry):
	"""Considers all points between two rectangles."""
	def __init__(self, x, y, width, height, thickness, inside_border=True, use_exact_coordinates=False):
		"""
		Use x, y, width, and height to specify a target rectangle.
		Then use thickness to say how much larger or smaller the second rectangle should me.
		If inside_border is True, this second rect will be placed inside the first.
		Otherwise, it will be placed around it.
		"""
		# Technically this doesn't matter. We're going to want all coordinate conversions
		# to be handled by the rectangles, so we'll override the default uses of this flag.
		# Nevertheless, set it properly in case someone decides to check it later.
		super().__init__(use_exact_coordinates)
		self.thickness, self.inside_border = thickness, inside_border
		# Having an inside rect just means that the secondary distances go in instead of out.
		# This can be handled by just inverting the thickness param
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
		# We've now created both rectangles.
		# Which one is on the inside depends on which way we adjusted the coordinates
		# to get the second rectangle (which depends on the sign of <thickness>)
		self.inner_rect, self.outer_rect = \
			(rect2, rect1) if thickness < 0 else (rect1, rect2)

	def _contains_point(self, x, y):
		return self.outer_rect._contains_point(x, y) and not \
			self.inner_rect._contains_point(x, y)

	# We want the rectangles to handle all coordinate conversions.
	# Override our own contains_raw_point function to never perform conversions
	def contains_raw_point(self, x, y, x_step, y_step):
		return self.outer_rect.contains_raw_point(x, y, x_step, y_step) and not \
			self.inner_rect.contains_raw_point(x, y, x_step, y_step)
