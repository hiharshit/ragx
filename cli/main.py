from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)

from ragx import rag_chain

app = typer.Typer(
    name="ragx",
    help="RAGx - End-to-end RAG pipeline with LangChain and Google Gemini",
    add_completion=False,
)
console = Console()


@app.command()
def ingest(
    path: Path = typer.Argument(..., help="Path to file or directory to ingest"),
) -> None:
    """Ingest documents into the vector store."""
    if not path.exists():
        console.print(f"[red]Error:[/red] Path does not exist: {path}")
        raise typer.Exit(1)

    console.print(f"[blue]Loading documents from {path}...[/blue]")

    try:
        count = rag_chain.ingest(
            path, on_progress=lambda msg: console.print(f"  [dim]{msg}[/dim]")
        )
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    console.print(f"[green]Success:[/green] Ingested {count} chunks from {path}")


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask"),
) -> None:
    """Query the RAG pipeline with a question."""
    stats = rag_chain.get_stats()
    if stats["document_count"] == 0:
        console.print(
            "[yellow]Warning:[/yellow] No documents indexed. Run 'ingest' first."
        )
        raise typer.Exit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Thinking...", total=None)
        try:
            answer = rag_chain.query(question)
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)

    console.print(Panel(answer, title="Answer", border_style="green"))


@app.command()
def chat() -> None:
    """Start an interactive chat session."""
    stats = rag_chain.get_stats()
    if stats["document_count"] == 0:
        console.print(
            "[yellow]Warning:[/yellow] No documents indexed. Run 'ingest' first."
        )
        raise typer.Exit(1)

    console.print(
        Panel(
            "RAGx Interactive Chat\nType 'exit' or 'quit' to end the session.",
            border_style="blue",
        )
    )

    while True:
        try:
            question = console.input("[bold cyan]You:[/bold cyan] ")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if question.lower() in {"exit", "quit", "q"}:
            console.print("[dim]Goodbye![/dim]")
            break

        if not question.strip():
            continue

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Thinking...", total=None)
            try:
                answer = rag_chain.query(question)
            except Exception as e:
                console.print(f"[red]Error:[/red] {e}")
                continue

        console.print(f"[bold green]RAGx:[/bold green] {answer}\n")


@app.command()
def clear() -> None:
    """Clear the vector store."""
    confirm = typer.confirm("Are you sure you want to clear all indexed documents?")
    if not confirm:
        console.print("[dim]Cancelled.[/dim]")
        raise typer.Exit(0)

    rag_chain.clear()
    console.print("[green]Success:[/green] Vector store cleared.")


@app.command()
def stats() -> None:
    """Show statistics about indexed documents."""
    data = rag_chain.get_stats()
    console.print(
        Panel(
            f"Documents indexed: {data['document_count']}",
            title="Stats",
            border_style="blue",
        )
    )


if __name__ == "__main__":
    app()
