#!/bin/sh
coverage run -m tests
coverage report -m
rm -f .coverage
# pyreverse -s2 -o png -p PyMut pymut
# pydoc -b
