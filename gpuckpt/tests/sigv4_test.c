/* Prints the Authorization header the C signer produces for the inputs on
 * the command line, so tests/run.sh can compare it with the Python
 * reference and with botocore. */
#include "objstore.h"
int main(int argc, char **argv)
{
    if (argc != 11) { fprintf(stderr, "usage: method host uri query payload_hash amz_date region ak sk token\n"); return 1; }
    char auth[1024];
    const char *token = argv[10][0] ? argv[10] : NULL;
    if (gc_s3_sign(argv[1], argv[2], argv[3], argv[4][0] ? argv[4] : NULL, argv[5], argv[6], argv[7], argv[8], argv[9], token, auth, sizeof auth))
        return 2;
    puts(auth);
    return 0;
}
