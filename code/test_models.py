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


def test_grid_from_coords():
    dense_ids, dense_long, dense_grid = models.grid_from_coords({
        "dense_A": [1, 2, 3],
        "dense_B": [10, 20],
    }, prefix="dense_")
    assert dense_ids.dims == ("dense_A", "dense_B")
    assert dense_long.dims == ("dense_id", "design_dim")
    assert dense_grid.dims == ("dense_A", "dense_B", "design_dim")
    assert tuple(dense_grid.design_dim.values) == ("A", "B")
    numpy.testing.assert_array_equal(dense_ids.values, numpy.array([
        [0, 1],
        [2, 3],
        [4, 5],
    ]))
    numpy.testing.assert_array_equal(dense_long.values, numpy.array([
        [1, 10],
        [1, 20],
        [2, 10],
        [2, 20],
        [3, 10],
        [3, 20],
    ]))
    numpy.testing.assert_array_equal(dense_grid.sel(design_dim="A").values, numpy.array([
        [1, 1],
        [2, 2],
        [3, 3],
    ]))
    numpy.testing.assert_array_equal(dense_grid.sel(design_dim="B").values, numpy.array([
        [10, 20],
        [10, 20],
        [10, 20],
    ]))
    pass
