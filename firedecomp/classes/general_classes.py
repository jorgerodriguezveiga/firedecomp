"""Module with general classes definition."""


# Element ---------------------------------------------------------------------
class Element(object):
    """Element class."""

    def __init__(self, name):
        """Element object.

        Args:
            name (:obj:`str` or `int`): element name.
        """
        self.name = name

    def get_index(self):
        """Return index."""
        return self.name

    def __repr__(self):
        index = self.name
        if isinstance(index, tuple):
            index_repr = ", ".join([i.__repr__() for i in index])
        else:
            index_repr = index.__repr__()
        return "<{}({})>".format(type(self).__name__, index_repr)
# --------------------------------------------------------------------------- #


# Set -------------------------------------------------------------------------
class Set(object):
    """Set class."""

    def __init__(self, elements):
        """Set of elements.

        Args:
            elements (:obj:`list`): list of elements.
        """
        self.__index__ = self.__check_names__(elements)
        self.elements = elements

    @staticmethod
    def __check_names__(elements):
        """Check resource names."""
        indices = {e.get_index(): i for i, e in enumerate(elements)}
        if len(elements) == len(indices):
            return indices
        else:
            raise ValueError("Element name is repeated.")

    def get_names(self):
        return self.__index__.keys()

    def get_element(self, *e):
        """Get element by name.

        Args:
            e (:obj:`str` or `int`): element name.
        """
        if len(e) == 1:
            e = e[0]
        if e in self.get_names():
            return self.elements[self.__index__[e]]
        else:
            raise ValueError("Unknown element name: '{}'".format(e))

    def size(self):
        return len(self.elements)

    def __iter__(self):
        return (e for e in self.elements)

    def __getitem__(self, element):
        return self.get_element(element)

    def __repr__(self):
        return self.elements.__repr__()
# --------------------------------------------------------------------------- #
