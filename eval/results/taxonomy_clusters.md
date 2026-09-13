# Taxonomy clusters (k=10)

8,000 customer messages from **AmazonHelp**, embedded with `sentence-transformers/all-MiniLM-L6-v2`, seed 20260909.

Committed so the human naming step in `src/threadpilot/taxonomy.py` is reviewable rather than asserted. Terms are *distinctive* (mean TF-IDF inside the cluster minus outside), not merely frequent - plain frequency surfaces 'order' and 'please' in every cluster and tells a reader nothing.

Far-from-centroid examples are included deliberately: cluster edges are where merge and split decisions get made, and reading only the centre makes every cluster look cleaner than it is.

## Cluster 3 - 994 msgs (12.4%), mean sim 0.6726

**Distinctive terms:** delivery, day, today, shipping, day delivery, ordered, date, day shipping, package, delivered, days, delivery date

**Nearest centroid:**

- <user> my order was scheduled to arrive yesterday and didnt. I was not informed that the parcel was late and could take up to 20 Oct to
- <user> i just placed an order with deliver next day by 1pm and its saying it wont arrive till monday?
- <user> <user> I ordered something via Amazon Prime midday on Friday 24th November and still have no goods. Tracker says “dispatched for delivery tomorrow” but has said this since Monday 27th November.
- <user> Once again Amazon, you have let me down by stating my order was delivered when it hasn't... Step up your game with your delivery services!
- <user> my order#408-5604067-5477926 supposed to be delivered by tomorrow but I have noticed it is not yet dispatched. Status ASAP
- <user> what's going on with deliveries? Ordered last Thursday with Prime, said guaranteed delivery Friday by 8pm, still not here today 😔
- <user> <user> trying to understand why I was guaranteed to get my package by the end of the day today &amp; paid for 2 day shipping yet somehow, Amazon is suddenly telling me my package won’t be deliv
- <user> I’ve ordered and paid for two day shipping, now you’re telling me the item is arriving Wed Nov 15th. Why?
- <user> I ordered something with prime shipping and it was supposed to come today, now it may not come until Tuesday?! Unacceptable. <url>
- Dear <user> my order #407-7208892-2195534 is suppose to deliver by 31 Oct 2017... Still not received...

**Far from centroid (cluster edge):**

- Seriously Lasership &amp; <user> : It’s 8:34PM. I go to bed @ 10pm. My stuff was loaded onto a truck @ 5:04 PM. If they wake me…..
- It took <user> 3 days to install washing machine. <user> waiting for 5 weeks. Make informed choices...
- <user> Will Xanathar's Guide to Everything be dispatched on time?
- Hi <user> if I've an order pending stock and the price of it drops before it's dispatched will I be charged the original or new lower price?

## Cluster 2 - 930 msgs (11.6%), mean sim 0.5409

**Distinctive terms:** refund, price, card, money, return, cashback, product, buy, charged, payment, bought, offer

**Nearest centroid:**

- See what <user> is doing not able to deliver the product admitt their mistake apolzs and inform returning your amount but not get it yet <url>
- <user> is joke. Seller arranged for my return item to be picked up and now I can’t get my refund. I’ve never experienced such poor service
- <user> is not refunding me since a month for order no. 4__credit_card__. Amazon is desparate to cheat customers. People pls avoid Amazon
- So, <user> doesn't want to refund my money? I have been patiently waiting. I sent the product back &amp; still haven't seen the money. Smh.
- <user> what d hell Amazon till now my refund was not provide very bad service by Amazon plz refund my balance
- . <user> haven’t gotten my refund from weeks ago. My order was cancelled, I want my money.
- <user> Promotional cash back not credited even after refund date been crossed <url> order# 404-4707947-1460330☹️😡
- And now <user> are only offering a partial refund and not the full amount...ahem, I never received any item at all
- <user> it's almost one month and refund not received. No response from their side. <url>
- <user> @ hi had purchased from amazon and i return back but its more then 1month i had not received my cash back...what to do ...

**Far from centroid (cluster edge):**

- hey <user> your tax interview counts me as a US person because I studied there for 2 years but I live in Sweden.. help :(
- <user> Is it possible to filter listings that have less than 50 reviews? I find that a ton of products have paid for people to leave 25-50 positive reviews.
- <user> hi, please could you tell me if the new COD ww2 game comes with pre order bonus, weapon unlock and 2xp please
- Even I said if you dnt have blue, I will take black but they says blue is not available n black they will not give..In replacement... <url>

## Cluster 6 - 907 msgs (11.3%), mean sim 0.6497

**Distinctive terms:** delivery, package, delivered, parcel, deliver, door, courier, user delivery, driver, house, packages, home

**Nearest centroid:**

- Dear <user> please do not use <user> for delivery service. They suck. They drove by my house three times without delivering on three separate days, yet still marked status as "delivered."
- <user> I've not received a parcel that's apparently been delivered?
- Nice. My <user> package was delivered to my neighbor. Good thing they just brought it over to me. Delivery person should be more careful.
- <user> i ordered a parcel that was supposed to be delivered on Tuesday, I had an email saying that it had been ‘delivered to resident’ at 4:03pm but i haven’t received it and there’s no note saying it
- <user> <user> The delivery guy number provided to me by you is not carrying my package. FUCKING service as always.
- <user> Your delivery driver missed a package they were supposed to deliver today.
- . <user> will you ever get a delivery right? Yet again delivery instructions have being ignored. Already contacted the head of logistics
- Wtf I’m so annoyed. My package from <user> says it was delivered yesterday and “handed to a resident” but I never got it. Now another package says it was delivered today and I don’t see it anywhere in
- The logistics/delivery for <user> is a joke, they just pushed through a 'sorry we missed you' card without even trying to knock. Another wasted day stuck at home waiting for a parcel that never came.
- Two <user> deliveries to my office already today but not my stuff. Beginning to think it may have fallen behind the delivery person's seat. #amazondelivery

**Far from centroid (cluster edge):**

- <user> AMZ Logistics: “No secure place to leave pkg” Building has mail room, desk &amp; I am home. How much more secure do you need?
- <user> This is happening far too often. I was in today and the driver did not ring the door bell or knock on the door. I saw him driving off. <url>
- <user> Package left, marked "heavy", steep driveway but plenty of room to turn around, green x is in front door area- ANYWHERE in that area would be fine but at 106lbs soaking wet package is too heavy
- <user> So, your driver doesn't get within an 8 minute drive of my house and marks my goods undeliverable.... with the baby food in his car. Cheers. Thanks to him, NONE of the drivers that come to my h

## Cluster 7 - 801 msgs (10.0%), mean sim 0.5391

**Distinctive terms:** url, user, account, user url, user user, help, email, dm, password, getting, user account, echo

**Nearest centroid:**

- <user> AGAIN not left in my safe place!! I’m getting really frustrated with this!! <url>
- <user> <user> thanks for your support
- <user> is a crap service and is falsely advertising they deliver asap. Shame on <user> <user>
- <user> <user> <user> <user> provided my details but <user> Is not ready to help <user> <url>
- <user> I did and it said that <user> won't be able to help me further with it. If you can't then who can? This is crazy!!😡 <url>
- <user> Still not got this ... <url>
- <user> can you give me more information than this I've been waiting in? All it's said since yesterday <url>
- <user> <user> <user> may I know why ? <url>
- <user> why can't I check out? <url>
- <user> <user> <user> <user> I tried returning it few weeks back also but they refused to take.Scamming customers.Shameful <url>

**Far from centroid (cluster edge):**

- <user> ordr no.1933 date 27-9 Ordrd book of civil <url> book of mech. Engg 5 rtrn requsts. Vry pur service nvr gonna by
- Is it standard procedure for <user> drivers to go peeking in open windows?
- Dear <user> , was just wondering when we declared the light as a safe place? I’m no hide and seek champion, but I’m guessing the driver isn’t one either! <url>
- <user> CADÊ MEUS LIVRO PORRA QUE AINDA NÃO FORAM NEM PRA TRANSPORTADORA

## Cluster 8 - 797 msgs (10.0%), mean sim 0.5031

**Distinctive terms:** url, packaging, box, ordered, thanks, gift, package, damaged, christmas, packing, user url, boxes

**Nearest centroid:**

- . <user> go from giving shit loads of packaging to nothing. And now the delivery stickers have ruined the packaging for a gift 🙄😡 <url>
- <user> WTF! Why would you ship my purchase like this!!! It was a gift #failed . <url>
- So <user> lost my package recently in transit &amp; had to buy a similar product from someone else. It's the 1st it's happened since with them. <url>
- <user> Guys bought a gift from the site. The shipping dept. Lost it after reaching destination replacement now needs a week more WTF :/
- <user> <user> disappointed to receive my parcel like this, especially with Christmas presents in which have been damaged <url>
- <user> Love the service but your packaging get worse by the day: this large box for this single item is ridiculous! <url>
- An outrageous amount of packaging for a small Christmas present <user> . Good job the <user> December issue has got a feature coming up on excess packaging! #OComeAllYeWasteful <url>
- Looks like <user> have sent my ordered item on a trip to the UK and back... #holidays <url>
- Hi, <user> is this how you package your items?! Absolutely atrocious! Sort it out! <url>
- <user> how my package was delivered. Not cool <url>

**Far from centroid (cluster edge):**

- <user> returns on swimsuits r nt allowd..agree 4m hygiene point of view. Bt if by luks, it is a kids wear in place of an adults?
- Hey <user> if you are going to drop wiper blades from S&amp;S maybe make sure the ones you suggest as a replacement fit my vehicle? <url>
- <user> Is there any way to set a preference to have you give dimensions in inches instead of centimetres?
- <user> purchased a stainless steel water bottle, the lip is rusting after 2 weeks...

## Cluster 9 - 787 msgs (9.8%), mean sim 0.6625

**Distinctive terms:** order, user order, received, id, 408, order id, order 408, delivered, cancel, 406, 171, 404

**Nearest centroid:**

- <user> please can you look into this order: 206-9303243-6219526 I paid for next day delivery and still not received!! I would like
- <user> i have just seen this order was cancelled and not by me. VERY UNHAPPY Order #701-5406600-7964269
- <user> Ordered on 24 September 2017 Order# 408-3424812-8857141 still not delivered no intimation pls help
- Hey <user> my order got delivered to someone else. Order Id: 171-6576778-6197962. Contacted customer care yesterday, NO Updates till now😟
- <user> My order 171-4078604-8109968 Is not been delivered to me, the committed date was 13th october 2017. Support is good for nothing.
- <user> Ordered on 7 October 2017 Order# 403-7284574-6433108 not yet delivered. No way to cancel and get refund.
- Hi <user> <user> I still didn't receive my below order, request you to please look into it... Order no: 404-1723987-5697113 <url>
- <user> pathetic service. My order got updated as delivered and never reached me. After waiting for 3 days as suggested by cust care
- <user> <user> Your customer Dept cannot provie exact status of my order. Pathetic . Hell bad .For human sake Help me Plz !
- <user> this is regarding my order# 171-5331733-1100309 which shows as delivered but in reality it's not delivered to me <user>

**Far from centroid (cluster edge):**

- <user> I ordered Sheffield Induction Base Cookware on 03-10-17. Which is defective, So I contacted your cust care via call almost everyday for 7 days. But the pick up haven't been happened as promised
- <user> <user> Your order for pin 800020 dispatches from Kolkata West Bengal to Haryana while it can directly be send to Patna.
- Defective water filter sold...the tap leaks in shut-position Order# 407-7562012-9054745 #Nuisance <user> <user> <user> <user>
- <user> when will the return pick up happen? When I called the given number, person says someone else will come but he doesn't know who! <url>

## Cluster 1 - 780 msgs (9.8%), mean sim 0.6434

**Distinctive terms:** amazon, user amazon, amazonindia, user user, amazon india, amazon pay, india, pay, account, amazongreatindianfestival, amazonprime, service amazon

**Nearest centroid:**

- <user> <user> <user> <user> <user> <user> <user> <user> . No repsonse from amazon yet. Pathetic services
- <user> <user> <user> What the hell Amazon is doing, fuck you Amazon. order 406-2675295-7486748
- Such a #PoorService given by Amazon. They never give us proper reply on call #pathetic Amazon <user> <user> #Amazonpitiful #Amazon <url>
- <user> your service is very unsatisfactory. me and my friends and family are also deeply unhappy by it. I'm never using Amazon again
- well done amazon you ditched me twice this week.I'm very frustrated this time <user> #amazonfails <url>
- Very pathetic service provider <user> ...be care full while buying any thing from Amazon. Very shame full #AmazonGreatIndianFestival
- <user> I'm sorry to say but calling Amazon is never easy
- <user> worst service ever from #Amazon same issue comming again and again and there is not solutions in customer care
- Another pathetic experience from Amazon india.not received my amazon pay cashback since 2 weeks <user> <user>
- <user> <user> <user> This is the real amazon. Think twice before using it. My money they cheated. Others please beware. Fraud <url>

**Far from centroid (cluster edge):**

- <user> are all LED bulbs/lamps sold on amazon BIS approved?
- <user> <user> <user> I have ordered oneplus 5t in amazon but the emi displayed in amazon portal is different and the emi scheme in axis website is different?Please acknowledge asap! Attached images fo
- <user> HI Amazon, where can I change my country of residence? Thanks
- <user> Amazon and I don't know what to do can you tell me how long is the warranty valid it was v amazing robust sturdy fine efficient

## Cluster 5 - 745 msgs (9.3%), mean sim 0.4306

**Distinctive terms:** app, alexa, kindle, echo, tv, music, video, stick, available, season, india, prime video

**Nearest centroid:**

- I would love to watch <user> but it’s “currently unavailable to watch in” my location. Please explain <user> <user> #tv #AmazonPrime #MarvelousMrsMaise
- <user> can you please sort out your <user> app, it’s dreadful. After watching adverts it says “content not available” 😡
- <user> <user> <user> When you are going to launch Amazon Prime Music in India ?
- <user> This app is not good please don't buy product with amazon it has not returned my money back
- <user> I can't run Amazon prime videos on my firetv stick without using Amazon prime app on my mobile. Why is that?
- <user> the netflix app isn't working on my fire stick but it's working on everyone else's. I've turned the internet off and on ..
- <user> Your Amazon Prime Music service will not stream on my Echo Device today.
- <user> Thanks for your prime service its top notch in both Odering in app and access to video . When you guys plan to launch prime music
- Why did i update my <user> app on my #amazon #firestick it won’t work now! Just sits loading then pops back out. Yes I rebooted the device. $25 aint worth it.
- <user> Alexa working well shame the free trial you get isn’t the same as what you pay for I can’t download Music for my echo

**Far from centroid (cluster edge):**

- <user> I won't able to attend my training days and I am going to need a shift change. To change my shift, I was told I'd have to go back to the Integrity Opportunity Center in either Edison or Linden,
- Check out the reviews by Students/Parents <user> for CBSE Question Bank for Class 12 Chemistry for March 2018 Exam. <url>
- Definition of a nanosecond: the time elapsed between the <user> delivery driver knocking on the door and writing a 'we missed you' card! 😬
- Ridley &amp; I have taken up yoga. <user> has made it easy so far, our beginners yoga DVD and mat was sent right to our door. Wish us luck! 🤞🐰 <url>

## Cluster 4 - 730 msgs (9.1%), mean sim 0.719

**Distinctive terms:** prime, user prime, amazon prime, day, membership, prime membership, shipping, days, pay, prime day, day shipping, prime delivery

**Nearest centroid:**

- <user> is disappointing. Couldn't even make the delivery on time...why spend money on prime to receive something days later!? 😡. <user>
- I don’t understand why <user> Prime isn’t shipping promptly anymore. I spend tens of thousands a year with them. No two day service. 😬
- Anyone else experiencing issues with <user> prime? Several of my orders over the last few weeks have taken more than 2 days for delivery. Why pay for this service?!
- <user> Used my Prime Membership again today for an order, not due to arrive until Tuesday, how can this be One Day delivery?! Paying Prime Membership for this poor service!!
- Oh look, another <user> Prime order that's gonna take 10 days to arrive... why am I paying for Prime again??
- <user> Is anyone else having problems with #prime? I pay for two days.... and lately my orders are getting here at three or four days. Why am I paying for two day shipping and not getting it?
- tbh <user> prime is useless until they handle their own shipping and can actually what I ordered to me in two days.
- <user> why do I pay for prime if my packages aren't here in 2 days?? I've had one order lost and another order was ordered on and showed up yesterday
- Stinks to pay for <user> prime when the last 3 orders have had shipping delays with items eligible for prime. 😐
- <user> this is literally like the fourth time a next day package has been delayed. Idk why i even pay for prime.

**Far from centroid (cluster edge):**

- <user> I am moving to Mexico, can i switch my prime account to Mexico?
- <user> pthetic std of prime srvices .nobody hs any idea whr d pkg is, dat I hv alrdy paid for . Shame amazon! #407-64432331483510
- <user> when renewing prime membership can I combine prime and music unlimited? Thanks.
- <user> hello there. I'm wondering if as Prime member, there is a cost to ship to Canada from the US?

## Cluster 0 - 529 msgs (6.6%), mean sim 0.6129

**Distinctive terms:** service, customer, customer service, user customer, worst, care, service user, support, customers, user user, customer care, phone

**Nearest centroid:**

- <user> your customer service sucks ass
- <user> why is your customer service so awful
- <user> not good customer service
- <user> customer service is woeful. No help whatsoever.
- <user> you're customer service sucks ass.
- <user> you guys need to work on your customer service cause I’ve had nothing but bad service for something so simple
- Been dealing with <user> customer service for the past week never would I have thought it would be this bad
- <user> has the worst customer service 🙄
- <user> trying to call customer service for last 2 hours. No response yet. Your service is really Poor
- WORST CUSTOMER SERVICE <user>

**Far from centroid (cluster edge):**

- <user> It sucks to be a flex driver get bit by a dog and be told well we don't guarantee safety. It would be courteous if someone reached out or even act like they cared. No just load me down with a h
- <user> If you could train your advisers to actually try and help and not just read out what I can already read myself on your tracking system that would be great. I don't really enjoy having to spell 
- <user> <user> <user> <user> <user> <user> can a police complaint be lodged online? <url>
- <user> bought iphone7 in july2017. Phone is defective, hangs, behaves crazy!! Battery drains in few hours only. How to get replacement?
