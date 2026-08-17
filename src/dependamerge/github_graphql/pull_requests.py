# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

"""
Open pull request queries.

The two queries here carry the full per-PR payload the scanner needs —
mergeability, base/head refs and repositories, changed files, recent
comments, reviews and the latest commit's status check rollup — so a scan
does not have to fall back to per-PR REST calls.

``ORG_REPOS_WITH_OPEN_PRS`` walks an organization repository-by-repository
with a first page of PRs attached; ``REPO_OPEN_PRS_PAGE`` continues a
single repository's PR pagination and takes its page sizes as variables.
"""

from __future__ import annotations

# Fetch organization repositories with a first page of their open PRs.
# Use the returned pageInfo to continue paging repositories.
# Each repository node also includes pageInfo for its pull requests; for repos
# with more than 50 open PRs, use REPO_OPEN_PRS_PAGE to paginate further.
ORG_REPOS_WITH_OPEN_PRS = """
query($org: String!, $reposCursor: String) {
  organization(login: $org) {
    repositories(first: 30, after: $reposCursor, orderBy: { field: NAME, direction: ASC }) {
      pageInfo {
        hasNextPage
        endCursor
      }
      nodes {
        nameWithOwner
        isArchived
        pullRequests(
          states: OPEN
          first: 30
          orderBy: { field: CREATED_AT, direction: DESC }
        ) {
          pageInfo {
            hasNextPage
            endCursor
          }
          nodes {
            id
            number
            title
            body
            url
            isDraft
            author { __typename login }
            mergeable
            mergeStateStatus
            baseRefName
            headRefName
            headRefOid
            headRepository { nameWithOwner url isFork }
            baseRepository { nameWithOwner url }
            createdAt
            updatedAt
            files(first: 50) {
              nodes {
                path
                additions
                deletions
              }
            }
            comments(first: 10, orderBy: { field: UPDATED_AT, direction: DESC }) {
              nodes {
                author { login }
                body
                createdAt
              }
            }
            reviews(first: 20, states: [PENDING, COMMENTED, APPROVED, CHANGES_REQUESTED]) {
              nodes {
                id
                author { login }
                state
                body
                createdAt
                updatedAt
              }
            }

            commits(last: 1) {
              nodes {
                commit {
                  oid
                  statusCheckRollup {
                    state
                    contexts(first: 20) {
                      nodes {
                        __typename
                        ... on CheckRun {
                          name
                          status
                          conclusion
                          startedAt
                          completedAt
                        }
                        ... on StatusContext {
                          context
                          state
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
"""

# Paginate open PRs for a specific repository when there are more than 50.
# Provide the repository owner/name and the PR cursor returned by previous pages.
REPO_OPEN_PRS_PAGE = """
query($owner: String!, $name: String!, $prsCursor: String, $prsPageSize: Int!, $filesPageSize: Int!, $commentsPageSize: Int!, $contextsPageSize: Int!) {
  repository(owner: $owner, name: $name) {
    nameWithOwner
    pullRequests(
      states: OPEN
      first: $prsPageSize
      after: $prsCursor
      orderBy: { field: CREATED_AT, direction: DESC }
    ) {
      pageInfo {
        hasNextPage
        endCursor
      }
      nodes {
        id
        number
        title
        body
        url
        isDraft
        author { __typename login }
        mergeable
        mergeStateStatus
        baseRefName
        headRefName
        headRefOid
        headRepository { nameWithOwner url isFork }
        baseRepository { nameWithOwner url }
        createdAt
        updatedAt
        files(first: $filesPageSize) {
          nodes {
            path
            additions
            deletions
          }
        }
        comments(first: $commentsPageSize, orderBy: { field: UPDATED_AT, direction: DESC }) {
          nodes {
            author { login }
            body
            createdAt
          }
        }
        reviews(first: 20, states: [PENDING, COMMENTED, APPROVED, CHANGES_REQUESTED]) {
          nodes {
            id
            author { login }
            state
            body
            createdAt
            updatedAt
          }
        }

        commits(last: 1) {
          nodes {
            commit {
              oid
              statusCheckRollup {
                state
                contexts(first: $contextsPageSize) {
                  nodes {
                    __typename
                    ... on CheckRun {
                      name
                      status
                      conclusion
                      startedAt
                      completedAt
                    }
                    ... on StatusContext {
                      context
                      state
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
"""
