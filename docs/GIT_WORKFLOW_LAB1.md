\# Git Version Control and Rollback Workflow



\## 1. Branch Management



Created separate Git branches for different project activities:



\- main

\- preprocessing

\- training

\- validation

\- deployment

\- experimentation



\## 2. Commit Management



Used meaningful commit messages to track project changes and maintain version history.



\## 3. Merge Conflict Resolution



Modified the same documentation file independently on the main and preprocessing branches. Git identified a merge conflict, which was resolved manually. The resolved changes were committed successfully.



\## 4. Rollback Demonstration



A temporary documentation file was created and committed to demonstrate rollback.



Original commit:

`77f4a00 Add temporary file for rollback practice`



Rollback command:



`git revert 77f4a00`



Revert commit:

`6abe58c Revert "Add temporary file for rollback practice"`



\## 5. Verification



\- Confirmed the temporary file was deleted.

\- Verified that both original and revert commits remained in Git history.

\- Confirmed that the working tree was clean.



\## Result



Git branching, committing, merging, conflict resolution, and safe rollback using `git revert` were successfully demonstrated.

