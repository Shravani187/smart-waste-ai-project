% Practical 2 — Prolog Facts, Rules and Queries
% Compatible with SWI-Prolog.
% Run with: swipl -s practical_02_family_and_waste_rules.pl
% Then enter queries at the ?- prompt.

% ===== Family relationship facts =====
% parent(Parent, Child)
parent(john, mary).
parent(john, tom).
parent(mary, sue).
parent(mary, bob).
parent(tom, ann).
parent(tom, kate).
parent(sue, leo).

% ===== Derived family rules =====
father(F, C) :- parent(F, C), male(F).
mother(M, C) :- parent(M, C), female(M).

male(john).
male(tom).
male(bob).
male(leo).
female(mary).
female(sue).
female(ann).
female(kate).

sibling(A, B) :- parent(P, A), parent(P, B), A \= B.

grandparent(GP, GC) :- parent(GP, P), parent(P, GC).

uncle(U, N) :- parent(P, N), sibling(U, P), male(U).
aunt(A, N) :- parent(P, N), sibling(A, P), female(A).

cousin(A, B) :- parent(PA, A), parent(PB, B), sibling(PA, PB), A \= B.

ancestor(A, D) :- parent(A, D).
ancestor(A, D) :- parent(A, X), ancestor(X, D).

% ===== Sample queries =====
% ?- father(john, mary).        -> true
% ?- sibling(mary, tom).        -> true
% ?- grandparent(john, sue).    -> true
% ?- uncle(tom, sue).           -> true
% ?- cousin(leo, ann).           -> true
% ?- ancestor(john, leo).       -> true

% ===== Waste-management reasoning =====
% waste_item(Name, DefaultCategory)
waste_item(apple_core, wet).
waste_item(food_scraps, wet).
waste_item(leaves, wet).
waste_item(newspaper, dry).
waste_item(clean_paper, dry).
waste_item(plastic_bottle, dry).
waste_item(glass_jar, dry).
waste_item(metal_can, dry).
waste_item(diaper, residual).
waste_item(broken_ceramic, residual).

% soiled(Item) - condition that makes paper non-recyclable
soiled(paper).
soiled(cardboard).

% categorize(Item, Category) with contamination rule
categorize(Item, wet) :- waste_item(Item, wet).
categorize(Item, dry) :- waste_item(Item, dry), \+ soiled(Item).
categorize(Item, residual) :- waste_item(Item, residual).
categorize(Item, residual) :- waste_item(Item, dry), soiled(Item).

% bin_status(Bin, FillLevel) and collection rule
bin_status(wet_bin, 85).
bin_status(dry_bin, 60).
bin_status(residual_bin, 90).

needs_collection(Bin) :- bin_status(Bin, Level), Level >= 80.

% ===== Waste sample queries =====
% ?- categorize(plastic_bottle, dry).  -> true
% ?- categorize(newspaper, residual).  -> true (soiled paper rule)
% ?- needs_collection(wet_bin).        -> true
% ?- needs_collection(dry_bin).        -> false
