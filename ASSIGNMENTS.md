# Publishing your solutions
In order for us to evaluate your solutions we rely on github classroom.

However, we have one common repo so you can keep everything at one place.

This however needs some trickery and we prepared a recipe for you.

First of all you need to clone the master repo:

  `git clone [git@github.com](mailto:git@github.com):automl-edu/RL-exercises.git`
  `cd RL-exercises`

(probably you have done this step if you have followed the installation instructions in `README.md`)

Now you have the skeleton for the individual exercises.

If we publish a new exercise then you need to click on the assignment link and accept it. Remember your repo, mine would be for week 1 `automl-edu/week-1-introduction-benjamc` .

The important thing is that you work in the master repository but push to the assignment repos. This is how it works:

You first add the remote like so:

`git remote add week1 git@github:automl-edu/week-1-introduction-benjamc.git` 

`week1` is the identifier of the remote and maybe you noticed the ID of the assignment repo.

After that, pull possible changes

`git pull week1 main --allow-unrelated-histories`

Maybe you need to merge and setup the merge:

1.  `git config pull.rebase false`
2. add all changes if necessary `git add *`
3. `git commit -m "Merge week1"`

You should be able to push to the new repo: `git push week1 main`

For the following weeks replace the number of the week remote identifier `week1` and your assignment repo.