#include <string.h>

void trim_newline(char *buf, int size) {
    for (int i = 0; i < size; i++) {
        if (buf[i] == '\n') {
            buf[i] = '\0';
            return;
        }
    }
}