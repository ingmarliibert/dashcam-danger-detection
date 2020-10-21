class Rectangle:
    def __init__(self, min_x=0,  min_y=0, max_x=0, max_y=0):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

    def is_intersect(self, other):
        if self.min_x > other.max_x or self.max_x < other.min_x:
            return False

        if self.min_y > other.max_y or self.max_y < other.min_y:
            return False

        return True

    def __and__(self, other):
        if not self.is_intersect(other):
            return Rectangle()
        min_x = max(self.min_x, other.min_x)
        max_x = min(self.max_x, other.max_x)
        min_y = max(self.min_y, other.min_y)
        max_y = min(self.max_y, other.max_y)
        return Rectangle(min_x, min_y, max_x, max_y)

    intersect = __and__

    def __or__(self, other):
        min_x = min(self.min_x, other.min_x)
        max_x = max(self.max_x, other.max_x)
        min_y = min(self.min_y, other.min_y)
        max_y = max(self.max_y, other.max_y)
        return Rectangle(min_x, min_y, max_x, max_y)

    union = __or__

    def __str__(self):
        return 'Rectangle({self.min_x},{self.max_x},{self.min_y},{self.max_y})'.format(self=self)

    @property
    def area(self):
        return (self.max_x - self.min_x) * (self.max_y - self.min_y)
