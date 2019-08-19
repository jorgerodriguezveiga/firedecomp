"""Module to write data classes information."""

# Python packages
import yaml
import logging as log


# write_data_class ------------------------------------------------------------
def write_data_class(data, file='data.yaml'):
    """Take a data object and write its attributes.

    Args:
        data: object with __dict__ attribute.
        file (:obj:`str`): filename.
    """
    log.debug('Write data')
    with open(file, 'w') as f:
        yaml.dump(data.__dict__, f, default_flow_style=False)
# --------------------------------------------------------------------------- #


# write_solution_as_csv -------------------------------------------------------
def write_solution_as_csv(
        data_dict, header=True, mode='w', file='data.csv', dec=".", sep=";"):
    """Take a data object and write its attributes.

    Args:
        data_dict (:obj:`dict`): dictionary.
        header (:obj:`bool`): if True write the header.
        mode (:obj:`str`): write mode: 'w' or 'a'.
        file (:obj:`str`): filename.
        dec (:obj:`str`): decimal point.
        sep (:obj:`str`): column separator
    """
    with open(file, mode) as f:
        if header:
            f.write(sep.join(data_dict.keys()) + '\n')

        f.write(sep.join([str(v).replace(".", dec)
                          if isinstance(v, (int, float)) else
                          v if v is not None else "None"
                          for v in data_dict.values()]) + '\n')
# --------------------------------------------------------------------------- #
