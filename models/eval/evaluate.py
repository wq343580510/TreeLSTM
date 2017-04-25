# -encoding:utf8-
import re


g_reP = re.compile(r"^[,?!:;]$|^-LRB-$|^-RRB-$|^[.]+$|^[`]+$|^[']+$|^（$|^）$|^、$|^。$|^！$|^？$|^…$|^，$|^；$|^／$|^：$|^“$|^”$|^「$|^」$|^『$|^』$|^《$|^》$|^一一$")

def eval(output, reference):
    total_uem = 1
    total = 0
    correct_head = 0
    correct_label = 0
    output = output.split("#")
    reference = reference.split("#")
    assert output[0] == reference[0]
    if len(output) != 3:
        return 0, 0, 0, 0
    if g_reP.match( output[0] ):
        return 0,0,0,0
    if output[1] == reference[1]:
        correct_head += 1
        if output[2] == reference[2]:
            correct_label += 1
    else:
        total_uem = 0
    total += 1
    return correct_head, correct_label, total, total_uem


def evaluate(lines , gold):
   total_sent = 0
   total_uem = 0
   total = 0
   correct_head = 0
   correct_label = 0
   for output,ref in zip(lines,gold):
      ret = eval(output, ref)
      correct_head += ret[0]
      correct_label += ret[1]
      total += ret[2]
      total_uem += ret[3]
      total_sent += 1
   #print float(correct_head) / total, float(correct_label) / total, float(total_uem) / total_sent
   return [float(correct_head) / total, float(correct_label) / total, float(total_uem) / total_sent]