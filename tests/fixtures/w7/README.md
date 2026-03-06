
# Password Validation Programs

## Overview

This repository contains two programs, `checkpasswd.c` and `validate.c`, which work together to validate user credentials.

## `checkpasswd.c`

- **Functionality**: Reads user ID and password from standard input.
- **Main Function**: `int main(void)`
  - Reads user ID and password using `fgets`.
  - TODO: Placeholder for further implementation.

## `validate.c`

- **Functionality**: Reads user ID and password from standard input, validates them, and checks against a password file.
- **Main Function**: `int main(void)`
  - Reads user ID and password using `read`.
  - Ensures null-termination and removes newline characters.
  - Concatenates user ID and password with a colon separator.
  - Opens the password file and checks for a match.
  - Exits with different status codes based on the result:
    - `0`: Match found.
    - `2`: Invalid password.
    - `3`: No such user.

## How They Work Together

1. `checkpasswd.c` reads the user ID and password.
2. `validate.c` processes the input, validates it, and checks it against the password file.
3. The results are communicated through exit status codes.

