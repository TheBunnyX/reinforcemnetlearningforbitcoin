from datetime import datetime

# Define the two datetime objects
dt1 = datetime(2023, 10, 19, 23, 1, 0)
dt2 = datetime(2023, 10, 19, 0, 1, 0)

# Calculate the difference in minutes
difference = (dt2 - dt1).total_seconds() / 60
print("Time difference in minutes:", difference)

# Corrected loop
loop = int(120 / 60)  # This will be 2, making it a range of 2
for i in range(loop):
    print("Number :", i)

# Print time in the format '19th October 2024, 12:00:00'
formatted_time_dt1 = dt1.strftime("%d %B %Y, %H:%M:%S")
formatted_time_dt2 = dt2.strftime("%d %B %Y, %H:%M:%S")
print("Formatted time dt1:", formatted_time_dt1)
print("Formatted time dt2:", formatted_time_dt2)
# Define the two datetime objects
add = 1
dt1 = datetime(2023, 10, 19, 15+1, 1, 0)
dt2 = datetime(2023, 10, 19, 16+1, 1, 0)
formatted_time_dt1 = dt1.strftime("%d %B %Y, %H:%M:%S")
formatted_time_dt2 = dt2.strftime("%d %B %Y, %H:%M:%S")
print("Formatted time dt1:", formatted_time_dt1)
print("Formatted time dt2:", formatted_time_dt2)

# 2023-09-16 23:58:00,6,26528.6,26529.7,26528.6,26529.8,0.25952227
# 2023-09-16 23:59:00,6,26529.8,26525,26525,26531,1.00055024
# 2023-09-17 00:00:00,0,26525,26522.1,26522.1,26525.1,1.78708444
# 2023-09-17 00:01:00,0,26522.2,26522.1,26522,26522.2,0.89355101