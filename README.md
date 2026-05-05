# CS-532-Final-Project
Github link: https://github.com/Lyle-Kottke/CS-532-Final-Project
### Dataset Column Definitions:
- VendorID : 1, 2, 6, 7 just companies providing taxi trip data
- tpep_pickup_datetime The date and time when the meter was engaged
- tpep_dropoff_datetime The date and time when the meter was disengaged.
- passenger_count The number of passengers in the vehicle.
- trip_distance The elapsed trip distance in miles reported by the taximeter.
- RatecodeID \
    The final rate code in effect at the end of the trip. (Preprogramed meter settings to determine fare) \

        1 = Standard rate 
        2 = JFK 
        3 = Newark 
        4 = Nassau or Westchester 
        5 = Negotiated fare 
        6 = Group ride 
        99 = Null/unknown 
- store_and_fwd_flag: This flag indicates whether the trip record was held in vehicle memory before sending to the vendor, aka“store and forward,” because the vehicle did not have a connection to the server. \

        Y = store and forward trip 
        N = not a store and forward trip 
- PULocationID: TLC Taxi Zone in which the taximeter was engaged.
- DOLocationID: TLC Taxi Zone in which the taximeter was disengaged.
- payment_type
    A numeric code signifying how the passenger paid for the trip. \

        0 = Flex Fare trip 
        1 = Credit card
        2 = Cash
        3 = No charge
        4 = Dispute
        5 = Unknown
        6 = Voided trip
- fare_amount: The time-and-distance fare calculated by the meter. For additional information on the following columns, see https://www.nyc.gov/site/tlc/passengers/taxi-fare.page 
- extra: Miscellaneous extras and surcharges.
- mta_tax: Tax that is automatically triggered based on the metered rate in use. -- Usually 0.5 dollars flat (as indicated in the data)
- tip_amount: Tip amount – This field is automatically populated for credit card tips. Cash tips are not included.
- tolls_amount: Total amount of all tolls paid in trip. (generally in longer trips)
- improvement_surcharge: Improvement surcharge assessed trips at the flag drop. The improvement surcharge began being levied in 2015.
- total_amount: The total amount charged to passengers. Does not include cash tips.
- congestion_surcharge: Total amount collected in trip for NYS congestion surcharge.
- airport_fee: For pick up only at LaGuardia and John F. Kennedy Airports. (generally longer, weakly correlated)
- cbd_congestion_fee: Per-trip charge for MTA's Congestion Relief Zone starting Jan. 5, 2025
