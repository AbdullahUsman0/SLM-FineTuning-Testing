# Explanation for the supervisor

## Professional English

Our model extracts the requirements needed by a tabular time-series forecasting pipeline. For example, a request to forecast temperature using hourly observations for the next three days contains a target, an observation frequency and a forecast horizon. The extractor maps these to the relevant schema slots. It does not itself predict the temperature. Likewise, the stocks and crypto examples measure requirement extraction, not investment advice or price forecasting accuracy.

We assess whether the model supplies the correct slot names, values and statuses, produces valid JSON, avoids adding unstated requirements, handles corrections and unknown information, and responds within a measured time. Exact F1 balances precision and recall over extracted requirements. A value of 0.8785 is not an 87.85% success rate for complete conversations; partial agreement can yield a high aggregate F1 while important state errors remain.

The earlier matched study used the same 31 development conversations for stock Qwen3.5-2B, v5, v6 and a hosted structured-output model. Rollout F1 was 0.4986 for v6 and 0.5086 for OpenAI. Their paired uncertainty interval included zero, so the evidence did not establish a winner or equivalence. V6 had higher precision and lower recall, which means it made a different tradeoff rather than being worse on every measure.

The later v7 pilot achieved 0.8785 rollout F1 versus v6's 0.8337 on a different 200-conversation synthetic development cohort. V7 nevertheless failed the promotion gate because extra slot flags increased from 64 to 72; mixed-language and dense requests also regressed. These scores cannot be directly compared with the earlier OpenAI score because the inputs and prompts differ. A matched v7–OpenAI comparison remains pending.

We have stopped further training and, because funds are limited, are using existing verified results for now. A new shared parser, private evaluation candidate, review materials and resumable evaluation code have been prepared offline. Independent human review has not been completed and no fresh final outcomes are claimed. No new paid API calls were made. This supports a transparent report about what the evidence shows and what it cannot yet establish.

## Easy Roman Urdu

Hamara model user ki forecasting requirements samajh kar schema ke slots fill karta hai. Misal ke taur par user kahe ke temperature ka forecast chahiye, observations hourly hain aur aglay teen din ka forecast chahiye. Model target, data ki frequency aur forecast horizon nikalta hai. Is experiment mein model khud temperature ya share ki price predict nahin karta.

Hum check karte hain ke slot ka naam, value aur status sahi hain; JSON valid hai; model apni taraf se extra requirement to nahin daal raha; correction aur “mujhe nahin pata” ko sahi handle karta hai; aur jawab kitni dair mein aata hai. F1 extracted requirements ka combined score hai. 87.85% F1 ka matlab yeh nahin ke 87.85% poori conversations bilkul sahi thin.

Purani matched testing mein same 31 conversations use hui thin. V6 ka rollout F1 49.86% aur OpenAI ka 50.86% tha. Uncertainty interval mein zero tha, is liye hum kisi ko clear winner nahin keh sakte. V6 ki precision zyada thi aur recall kam thi: us ne kam requirements pakri, lekin jo pakri un mein proportion ke hisaab se zyada sahi thin.

V7 pilot ka score doosray 200 synthetic conversations par 87.85% tha, jab ke usi test mein v6 ka 83.37% tha. V7 mein extra slot flags 64 se 72 ho gaye, is liye promotion gate pass nahin hua. Mixed-language aur dense requests par bhi score kam hua. V7 ka 87.85% aur puranay OpenAI test ka 50.86% seedha compare karna fair nahin: dono tests ke cases aur prompts alag hain.

Funds bachanay ke liye filhaal hum puranay verified results use kar rahe hain. Koi nayi paid API call ya training start nahin ki. Naya evaluation setup offline tayyar hai, lekin independent human review aur matched v7–OpenAI test abhi pending hain. Report mein hum yeh baat wazeh rakhein ge aur results se barh kar claim nahin karein ge.
