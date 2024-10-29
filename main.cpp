


#include "tokenizer.h"
#include <unistd.h>

int  main(int argc, char **argv){
    int opt = 0;
    std::string tiktoken_conf;
    while ((opt = getopt(argc, argv, "t:")) != -1) {
        switch (opt) {
        case 't':
            tiktoken_conf = optarg;
            break;

        default:
            std::cerr << "unknown command option" << std::endl;
            return -1;
            break;
        }
    }

    std::string config_path = tiktoken_conf;
    tokenizer::TiktokenFactory factory(config_path); 
    std::vector<int> id;
    std::string decode_string;
    factory.encode("hello world",id);
    if (factory.decode(id) != "hello world") {
        std::cout << "failed to test tiktoken encode and decode" <<  decode_string << std::endl;
        return -1;
    }

}
 
    