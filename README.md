# Exploratory data analysis: Customer loans
# 



# Set up notes
There is a package conflict with missingno and matplotlib that needs to be fixed before running.
After creating the environment, open:
>>~/anaconda3/envs/aicore_eda/lib/python3.12/site-packages/missingno/missingno.py
Find and replace all three instances of
grid(b=False)
with 
grid(visible=false)
Solution here:
https://stackoverflow.com/questions/35970686/ansible-ssh-error-unix-listener-too-long-for-unix-domain-socket/35971053#35971053