"""Module with general classes definition."""

# Python packages
import copy


# Element ---------------------------------------------------------------------
class Element(object):
    """Element class."""

    def __init__(self, name):
        """Element object.

        Args:
            name (:obj:`str` or `int`): element name.
        """
        self.name = name

    def copy(self, deep=False):
        if deep:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)

    def get_index(self):
        """Return index."""
        return self.name

    def update(self, dictionary):
        """Update object attributes.

        Args:
            dictionary (:obj:`dict`): dictionary with attribute information to
                update.
        """
        self.__dict__.update(dictionary)

    # def __setattr__(self, key, value):
    #     if key.startswith("__"):
    #         raise AttributeError(
    #             "Not allow assignment to protected member '{}'".format(key))
    #     self.__dict__[key] = value

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

    @staticmethod
    def __check_names__(elements):
        """Check resource names."""
        indices = {e.get_index(): e for e in elements}
        if len(elements) == len(indices):
            return indices
        else:
            raise ValueError("Element name is repeated.")

    def get_names(self):
        return list(self.__index__.keys())

    def get_element(self, *e):
        """Get element by name.

        Args:
            e (:obj:`str` or `int`): element name.
        """
        if len(e) == 1:
            e = e[0]
        if e in self.get_names():
            return self.__index__[e]
        else:
            raise ValueError("Unknown element name: '{}'".format(e))

    def get_info(self, attr):
        """Get a dictionary with attribute information of the elements.

        Args:
            attr (:obj:`str`): attribute name.
        """
        return {k: getattr(e, attr)
                for k, e in self.__index__.items()}

    def size(self):
        return len(self.__index__)

    def copy(self, deep=True):
        if deep:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)

    def update(self, dictionary):
        """Update object attributes.

        Args:
            dictionary (:obj:`dict`): dictionary with attribute information to
                update.
        """
        for e, v in dictionary.items():
            self.get_element(e).__dict__.update(v)

    # def __setattr__(self, key, value):
    #     if key.startswith("__"):
    #         raise AttributeError(
    #             "Not allow assignment to protected member '{}'".format(key))
    #     self.__dict__[key] = value

    def add(self, element):
        new_id = element.get_index()
        if new_id not in self.get_names():
            self.__index__[new_id] = element

    def __iter__(self):
        return (e for e in self.__index__.values())

    def __getitem__(self, element):
        return self.get_element(element)

    def __setitem__(self, key, value):
        self.__index__[key] = value

    def __repr__(self):
        return [e for e in self].__repr__()
# --------------------------------------------------------------------------- #
