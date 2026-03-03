
# README.md

## Overview

This repository contains two C programs: `checkpasswd.c` and `validate.c`. These programs are designed to handle user authentication by reading user IDs and passwords from standard input and checking them against a password file.

## `checkpasswd.c`

### Description

`checkpasswd.c` is a simple program that reads a user ID and a password from standard input. It uses the `fgets` function to read these inputs into the `user_id` and `password` arrays. If any read operation fails, it prints an error message and exits with a status of 1.

### Code

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include "common.h"

int main(void) {
  char user_id[MAXLINE];
  char password[MAXLINE];

  if(fgets(user_id, MAXLINE, stdin) == NULL) {
      perror("fgets");
      exit(1);
  }
  if(fgets(password, MAXLINE, stdin) == NULL) {
      perror("fgets");
      exit(1);
  }
  
  return 0;
}
```

## `validate.c`

### Description

`validate.c` is a more complex program that reads a user ID and a password from standard input, processes them, and checks if they match a user ID and password pair in a password file. The program uses the `read` function to read the inputs into the `userid` and `password` arrays. It ensures that the inputs are null-terminated and removes any newline characters. It then concatenates the user ID and password with a colon separator and checks this concatenated string against each line in the password file.

### Code

```c
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

// NOTE: MAX_PASSWORD must be less than  MAXLINE/2
#define MAXLINE 32
#define MAX_PASSWORD 10  

#define PASSWORD_FILE "pass.txt"

/* Reads two chunks from stdin, and checks if they match a user id
 * and password pair from a password file. The first chunk (MAX_PASSWORD bytes)
 * will contain a user id, and the second chunk (MAX_PASSWORD bytes) will contain 
 * a password.
 * 
 * The program exits with a value of 
 *      0 if the user id and password match,
 *      1 if there is an error, 
 *      2 if the user id is found but the password does not match, and 
 *      3 if the user id is not found in the password file. 
 */

int main(void){
    int n, user_length;
    char userid[MAXLINE];
    char password[MAXLINE];

    if ((n = read(STDIN_FILENO, userid, MAX_PASSWORD)) == -1) {
        perror("read");
        exit(1);
    } else if(n == 0) {
        fprintf(stderr, "Error: could not read from stdin");
        exit(1);
    }

    // Make sure user id is null-terminated
    if(n <= MAX_PASSWORD) {
        userid[n] ='\0';
    }

    // Remove newline character if it exists
    char *newline;
    if((newline=strchr(userid, '
')) != NULL) {
        *newline = '\0';
    }

    if ((n = read(STDIN_FILENO, password, MAX_PASSWORD)) == -1) {
        perror("read");
        exit(1);
    } else if (n == 0) {
        fprintf(stderr, "Error: could not read from stdin");
        exit(1);
    }

    // Make sure password is null-terminated
    if(n <= MAX_PASSWORD) {
        password[n] ='\0';
    }

    // Remove newline character if it exists
    if((newline=strchr(password, '
')) != NULL) {
        *newline = '\0';
    }

    // We expect userid to have enough space to concatenate ":" + password
    // but we will play it safe and use strncat
    strncat(userid, ":", MAXLINE - strlen(userid) - 1);
    user_length = strlen(userid);
    strncat(userid, password, MAXLINE - strlen(userid) - 1);

    FILE *fp = fopen(PASSWORD_FILE, "r");
    if (!fp) {
        perror("fopen");
        exit(1);
    }
    char line[MAXLINE];
    while(fgets(line, sizeof(line) - 1, fp)) {
        line[strlen(line) - 1] = '\0';
        if (strcmp(userid, line) == 0) {
            exit(0); // found match
        }
        else if(strncmp(userid, line, user_length) == 0)
            exit (2); // invalid password
    }
    exit(3); // no such user
}
```

## How They Work Together

1. **Input Handling**: Both programs read user ID and password from standard input. `checkpasswd.c` uses `fgets`, while `validate.c` uses `read`.

2. **Processing**: `validate.c` processes the inputs by ensuring they are null-terminated and removing any newline characters. It then concatenates the user ID and password with a colon separator.

3. **Validation**: `validate.c` checks the concatenated user ID and password against each line in the password file (`pass.txt`). It exits with different status codes based on whether the user ID and password match, the user ID is found but the password does not match, or the user ID is not found.

4. **Error Handling**: Both programs handle errors during input reading and file operations, printing error messages and exiting with appropriate status codes.

## Conclusion

These programs provide a basic framework for user authentication. `checkpasswd.c` reads the inputs, while `validate.c` processes and validates them against a password file. Together, they demonstrate fundamental input handling, string manipulation, and file operations in C.
