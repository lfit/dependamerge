# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

"""
Merge policy query and auto-merge mutation.

``GET_BRANCH_PROTECTION`` reports what a repository and its target branch
will actually accept — which merge methods are enabled, and which
protection rules must be satisfied first — and ``ENABLE_AUTO_MERGE`` is
the mutation used once those requirements are known to be reachable but
not yet met.
"""

from __future__ import annotations

# GraphQL mutation to enable auto-merge on a pull request
ENABLE_AUTO_MERGE = """
mutation EnableAutoMerge($pullRequestId: ID!, $mergeMethod: PullRequestMergeMethod) {
  enablePullRequestAutoMerge(input: {
    pullRequestId: $pullRequestId
    mergeMethod: $mergeMethod
  }) {
    pullRequest {
      autoMergeRequest {
        enabledAt
        enabledBy { login }
        mergeMethod
      }
    }
  }
}
"""

# GraphQL query to get branch protection settings for a repository
GET_BRANCH_PROTECTION = """
query GetBranchProtection($owner: String!, $name: String!, $branch: String!) {
  repository(owner: $owner, name: $name) {
    mergeCommitAllowed
    squashMergeAllowed
    rebaseMergeAllowed
    ref(qualifiedName: $branch) {
      branchProtectionRule {
        requiresLinearHistory
        requiresCommitSignatures
        requiredStatusCheckContexts
        requiresStatusChecks
        requiresApprovingReviews
        requiredApprovingReviewCount
        dismissesStaleReviews
        requiresCodeOwnerReviews
        restrictsPushes
        restrictsReviewDismissals
      }
    }
  }
}
"""
