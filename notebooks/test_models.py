import numpy
import xarray

import models


def test_reshape_dim():
    long = xarray.DataArray(
        [1,2,3,4,5,6],
        dims=("long",),
        coords={"long": list("ABCDEF")}
    )
    dense = xarray.DataArray(
        [[1,2],[3,4],[5,6]],
        dims=("ver","hor"),
        coords={"ver": list("ACE"), "hor": list("AB")}
    )
    new = models.reshape_dim(
        var=long,
        name="new",
        from_dim="long",
        to_dims=("ver", "hor"),
        to_shape=(3, 2),
        coords=dense.coords
    )
    numpy.testing.assert_array_equal(dense.values, new.values)
    numpy.testing.assert_array_equal(dense.sel(ver="C").values, new.sel(ver="C").values)
    pass
