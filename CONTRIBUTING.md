# Contributing to PySEMTools
Please read the following guide before contributing new code or a bug fix to PySEMTools.

All contributions to Neko must be made under the 3-Clause BSD license.

## Git branches
PySEMTools loosely follows the Git branching model described in https://nvie.com/posts/a-successful-git-branching-model, where `main` (in place of `develop`) contains the latest contributions, and all pull requests should start from `main` and be merged back into `main`.  New branches should be named `feature/<description>` for new features or `fix/<description>` for bug fixes.

When a pull request is submitted, a series of continuous integration tests will be run. A pull request will not be accepted nor merged into `main` until it passes the test suite.

## Code style
We use the `black` formatter with default settings. Therefore, new code added should simply be passed thorugh it. We have not made automatic actions in github to perform this task.

A linter is part of the CI test and might fail if correct formatting is not kept.

## Reporting issues
Bugs and issues with the code should be reported through Github issues, likewise, request for support are also managed through Github Issues.