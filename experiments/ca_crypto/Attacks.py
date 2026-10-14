from enc_dec import Encrypt,Decrypt
from converter import Conv
from Crypt_CA_1 import *

enc_bin = encrypted_binary
dec_bin = bin_string

enc_bin = enc_bin[1:-1]

enc = Encrypt()
dec = Decrypt()

c = Conv()


def generate_rule(rule_number):
    rule_dict = {}
    binary_repr = format(rule_number, "08b")
    for i in range(7, -1, -1):
        pattern = tuple(int(bit) for bit in format(i, "03b"))
        rule_dict[pattern] = int(binary_repr[7 - i])
    return rule_dict

def get_prev_generation_array(enc_bin,rule_dic):

    all_poss_decrp = []
    for i in range(0,len(enc_bin)):
        prev_block_poss = [''.join([str(k) for k in key]) for key in rule_dic.keys() if rule_dic[key] == int(enc_bin[i])]

        if i==0:
            itm = [k for k in prev_block_poss if k[0]=='0']
            all_poss_decrp = [itm[k][1:] for k in range(len(itm))]

        elif i==len(enc_bin)-1:
            itm = [k for k in prev_block_poss if k[-1]=='0']
            all_poss_decrp = [k for k in all_poss_decrp if k[-2:]+'0' in itm]
    
        else:
            bin3 = []
            for j in range(len(all_poss_decrp)):
                itm = [k for k in prev_block_poss if k[:-1]==all_poss_decrp[j][-2:]]
                if itm!=[]:
                    bin3 += [all_poss_decrp[j]+k[-1] for k in itm]

            all_poss_decrp = bin3

    return all_poss_decrp

def recur_brute_force(enc_bin):
    all_poss_bin_tree = [enc_bin]


for rule in range(1,257):
    rule_dic = generate_rule(rule)

    gen = generations

    all_poss_bin = [enc_bin]
    brute_force = []
    new_gen = []
    count = 0

    for t in range(gen):
        s = []
        for k in all_poss_bin:
            s += get_prev_generation_array(k,rule_dic)
        
        new_gen += [s]

        all_poss_bin = new_gen[count]
        count+=1

        if t==generations-2:
            break

    print(all_poss_bin)
    txt = []

    for i in range(len(all_poss_bin)):
        txt.append(c.convert_binary_to_text('0'+all_poss_bin[i]+'0'))

    print(txt)
    
    break
