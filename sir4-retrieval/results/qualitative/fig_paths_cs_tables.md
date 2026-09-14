## A. Route composition of the top path, by graph-channel outcome

SciAffordGraph + CCMP | same | graph rank \leq5 | n=86 | direct typed link 38%, via function/limitation 22%, via method/task frames 14%, via domain hub 24%, no valid path 1%
SciAffordGraph + CCMP | same | 6 to 50 | n=75 | direct typed link 8%, via function/limitation 9%, via method/task frames 15%, via domain hub 64%, no valid path 4%
SciAffordGraph + CCMP | same | >50 | n=76 | via function/limitation 7%, via method/task frames 3%, via domain hub 76%, no valid path 14%
OpenIE graph | same | graph rank \leq5 | n=42 | direct entity mention 100%
OpenIE graph | same | 6 to 50 | n=23 | direct entity mention 91%, co-mention chain 9%
OpenIE graph | same | >50 | n=170 | direct entity mention 18%, co-mention chain 81%, no valid path 1%
SciAffordGraph + CCMP | cross | graph rank \leq5 | n=162 | direct typed link 35%, via function/limitation 25%, via method/task frames 17%, via domain hub 16%, no valid path 7%
SciAffordGraph + CCMP | cross | 6 to 50 | n=121 | direct typed link 3%, via function/limitation 13%, via method/task frames 10%, via domain hub 62%, no valid path 12%
SciAffordGraph + CCMP | cross | >50 | n=210 | via function/limitation 9%, via method/task frames 3%, via domain hub 68%, no valid path 20%
OpenIE graph | cross | graph rank \leq5 | n=93 | direct entity mention 100%
OpenIE graph | cross | 6 to 50 | n=51 | direct entity mention 86%, co-mention chain 14%
OpenIE graph | cross | >50 | n=349 | direct entity mention 14%, co-mention chain 76%, no valid path 10%

## B. Hop length of the top path (valid paths only)

SciAffordGraph + CCMP | same | golds 237 | valid top path 222 | mean hops 2.73 | shortest seed->gold mean 2.21 | unreachable within 6: 0
SciAffordGraph, no CCMP | same | golds 237 | valid top path 222 | mean hops 2.67 | shortest seed->gold mean 2.20 | unreachable within 6: 0
OpenIE graph | same | golds 235 | valid top path 234 | mean hops 2.70 | shortest seed->gold mean 2.23 | unreachable within 6: 1
SciAffordGraph + CCMP | cross | golds 493 | valid top path 424 | mean hops 2.95 | shortest seed->gold mean 2.50 | unreachable within 6: 5
SciAffordGraph, no CCMP | cross | golds 493 | valid top path 426 | mean hops 2.91 | shortest seed->gold mean 2.51 | unreachable within 6: 5
OpenIE graph | cross | golds 493 | valid top path 458 | mean hops 2.70 | shortest seed->gold mean 2.24 | unreachable within 6: 35

## C. Graph-channel rank vs fused rank (first arm)

SciAffordGraph + CCMP | same | golds 237 | graph<=5 86 | of which fused<=5 80, fused>5 6 | graph<=5 while dense>5 14 | medians graph 13 fused 4 dense 4
SciAffordGraph + CCMP | cross | golds 493 | graph<=5 162 | of which fused<=5 148, fused>5 14 | graph<=5 while dense>5 27 | medians graph 25 fused 4 dense 4
