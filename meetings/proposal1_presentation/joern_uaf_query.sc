// Joern query: find Use-After-Free candidates
// Pattern: free(ptr) where ptr has an alias that is not nullified afterward

// Step 1: find all free()-like calls
val freeCalls = cpg.call.name(".*free.*").l

// Step 2: for each free call, get the freed identifier
val uafCandidates = cpg.call.name(".*free.*")
  .where(_.argument(1).isIdentifier)
  .map { freeCall =>
    val freedVar = freeCall.argument(1).code
    val method = freeCall.method

    // Step 3: find assignments where this variable is stored into another location
    // (i.e., aliases like sock->sk = sk)
    val aliases = method.assignment
      .where(_.source.isIdentifier.name(freedVar))
      .target.code.l

    // Step 4: check if any alias is nullified AFTER the free
    val freeLineNo = freeCall.lineNumber.get
    val nullifiedAfterFree = method.assignment
      .where(_.lineNumber.map(_ > freeLineNo))
      .where(_.source.code("NULL|nullptr|0"))
      .target.code.l

    // Step 5: aliases NOT nullified = potential UAF
    val danglingAliases = aliases.filterNot(a => nullifiedAfterFree.contains(a))

    (freeCall.code, freeCall.lineNumber, danglingAliases)
  }
  .filter(_._3.nonEmpty)

uafCandidates.foreach { case (code, line, aliases) =>
  println(s"[UAF] $code at line ${line.getOrElse("?")} — dangling aliases: ${aliases.mkString(", ")}")
}
