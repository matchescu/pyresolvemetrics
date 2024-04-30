# PyResolveMetrics

This is a standards compliant ([PEP-544](https://peps.python.org/pep-0544/)) and
efficient library for computing entity resolution metrics.
The library currently supports two ways of modelling entity resolution:

* [Fellegi-Sunter](https://www.tandfonline.com/doi/abs/10.1080/01621459.1969.10501049)
* [algebraic](https://www.igi-global.com/chapter/information-quality-management/23022)

## Set up

This library uses [Poetry](https://python-poetry.org) to manage dependencies and
`make` for build management. Please make sure these tools are installed.

## Run tests

The library has some unit tests that you can run to verify whether it's
operational.

```shell
$ make test
```

## Usage sample

Sample code that's informative about the library's capabilities
[is available](./sample/sample.py).
Check it out to figure out how the library works.
The [unit tests](./tests/) also double as documentation. 
